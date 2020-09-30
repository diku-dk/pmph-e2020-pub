-- BFAST-irregular: version handling obscured observations (e.g., clouds)
-- ==
-- compiled input @ data/sahara.in.gz
-- compiled input @ data/peru.in.gz

-- output @ data/peru.out.gz

let logplus (x: f32) : f32 =
  if x > (f32.exp 1)
  then f32.log x else 1

let adjustValInds [N] (n : i32) (ns : i32) (Ns : i32) (val_inds : [N]i32) (ind: i32) : i32 =
    if ind < Ns - ns then (val_inds[ind+ns]) - n else -1

let filterPadWithKeys [n] 't
           (p : (t -> bool))
           (dummy : t)           
           (arr : [n]t) : ([n](t,i32), i32) =
  let tfs = map (\a -> if p a then 1 else 0) arr
  let isT = scan (+) 0 tfs
  let i   = last isT
  let inds= map2 (\a iT -> if p a then iT-1 else -1) arr isT
  let rs  = scatter (replicate n dummy) inds arr
  let ks  = scatter (replicate n 0) inds (iota n)
  in  (zip rs ks, i) 

-- | builds the X matrices; first result dimensions of size 2*k+2
let mkX_with_trend [N] (k2p2: i32) (f: f32) (mappingindices: [N]i32): [k2p2][N]f32 =
  map (\ i ->
        map (\ind ->
                if i == 0 then 1f32
                else if i == 1 then r32 ind
                else let (i', j') = (r32 (i / 2), r32 ind)
                     let angle = 2f32 * f32.pi * i' * j' / f 
                     in  if i % 2 == 0 then f32.sin angle 
                                       else f32.cos angle
            ) mappingindices
      ) (iota k2p2)

let mkX_no_trend [N] (k2p2m1: i32) (f: f32) (mappingindices: [N]i32): [k2p2m1][N]f32 =
  map (\ i ->
        map (\ind ->
                if i == 0 then 1f32
                else let i = i + 1
		     let (i', j') = (r32 (i / 2), r32 ind)
                     let angle = 2f32 * f32.pi * i' * j' / f 
                     in  if i % 2 == 0 then f32.sin angle 
                                       else f32.cos angle
            ) mappingindices
      ) (iota k2p2m1)

---------------------------------------------------
-- Adapted matrix inversion so that it goes well --
-- with intra-blockparallelism                   --
---------------------------------------------------

  let gauss_jordan [nm] (n:i32) (m:i32) (A: *[nm]f32): [nm]f32 =
    loop A for i < n do
      let v1 = A[i]
      let A' = map (\ind -> let (k, j) = (ind / m, ind % m)
                            in if v1 == 0.0 then A[k*m+j] else
                            let x = (A[j] / v1) in
                                if k < n-1  -- Ap case
                                then ( A[(k+1)*m+j] - A[(k+1)*m+i] * x )
                                else x      -- irow case
                   ) (iota nm)
      in  scatter A (iota nm) A'

  let mat_inv [n] (A: [n][n]f32): [n][n]f32 =
    let m = 2*n
    let nm= n*m
    -- Pad the matrix with the identity matrix.
    let Ap = map (\ind -> let (i, j) = (ind / m, ind % m)
                          in  if j < n then ( A[i,j] )
                                       else if j == n+i
                                            then 1.0
                                            else 0.0
                 ) (iota nm)
    let Ap' = gauss_jordan n m Ap
    -- Drop the identity matrix at the front!
    in ((unflatten n m Ap')[0:n,n:m]) :> [n][n]f32
--------------------------------------------------
--------------------------------------------------

let dotprod [n] (xs: [n]f32) (ys: [n]f32): f32 =
  reduce (+) 0.0 <| map2 (*) xs ys

let matvecmul_row [n][m] (xss: [n][m]f32) (ys: [m]f32) =
  map (dotprod ys) xss

let dotprod_filt [n] (vct: [n]f32) (xs: [n]f32) (ys: [n]f32) : f32 =
  f32.sum (map3 (\v x y -> x * y * if (f32.isnan v) then 0.0 else 1.0) vct xs ys)

let matvecmul_row_filt [n][m] (xss: [n][m]f32) (ys: [m]f32) =
    map (\xs -> map2 (\x y -> if (f32.isnan y) then 0 else x*y) xs ys |> f32.sum) xss

let matmul_filt [n][p][m] (xss: [n][p]f32) (yss: [p][m]f32) (vct: [p]f32) : [n][m]f32 =
  map (\xs -> map (dotprod_filt vct xs) (transpose yss)) xss

----------------------------------------------------
----------------------------------------------------

-- | implementation is in this entry point
--   the outer map is distributed directly
entry main [m][N] (trend: i32) (k: i32) (n: i32) (freq: f32)
                  (hfrac: f32) (lam: f32)
                  (mappingindices : [N]i32)
                  (images : [m][N]f32) =
  ----------------------------------
  -- 1. make interpolation matrix --
  ----------------------------------
  let k2p2 = 2*k + 2
  let k2p2' = if trend > 0 then k2p2 else k2p2-1
  let X = intrinsics.opaque <|
	  if trend > 0
          then mkX_with_trend k2p2' freq mappingindices
	  else mkX_no_trend   k2p2' freq mappingindices
  

  -- PERFORMANCE BUG: instead of `let Xt = copy (transpose X)`
  --   we need to write the following ugly thing to force manifestation:
  let zero = r32 <| (N*N + 2*N + 1) / (N + 1) - N - 1
  let Xt  = intrinsics.opaque <|
            map (map (+zero)) (copy (transpose X))

  let Xh  =  (X[:,:n])
  let Xth =  (Xt[:n,:])
  let Yh  =  (images[:,:n])
  
  ----------------------------------
  -- 2. mat-mat multiplication    --
  ----------------------------------
  let Xsqr = intrinsics.opaque <|
             map (matmul_filt Xh Xth) Yh

  ----------------------------------
  -- 3. matrix inversion          --
  ----------------------------------
  let Xinv = intrinsics.opaque <|
             map mat_inv Xsqr
  ---------------------------------------------
  -- 4. several matrix-vector multiplication --
  ---------------------------------------------
  let beta0  = map (matvecmul_row_filt Xh) Yh   -- [2k+2]
               |> intrinsics.opaque

  let beta   = map2 matvecmul_row Xinv beta0    -- [2k+2]
               |> intrinsics.opaque -- ^ requires transposition of Xinv
                                    --   unless all parallelism is exploited

  let y_preds= map (matvecmul_row Xt) beta      -- [N]
               |> intrinsics.opaque -- ^ requires transposition of Xt (small)
                                    --   can be eliminated by passing
                                    --   (transpose X) instead of Xt

  ---------------------------------------------
  -- 5. filter etc.                          --
  ---------------------------------------------
  let (Nss, y_errors, val_indss) = ( intrinsics.opaque <| unzip3 <|
    map2 (\y y_pred ->
            let y_error_all = zip y y_pred |>
                map (\(ye,yep) -> if !(f32.isnan ye) 
                                  then ye-yep else f32.nan )
            let (tups, Ns) = filterPadWithKeys (\y -> !(f32.isnan y)) (f32.nan) y_error_all
            let (y_error, val_inds) = unzip tups
            in  (Ns, y_error, val_inds)
         ) images y_preds )

  ------------------------------------------------
  -- 6. ns and sigma (can be fused with above)  --
  ------------------------------------------------
  let (hs, nss, sigmas) = intrinsics.opaque <| unzip3 <|
    map2 (\yh y_error ->
            let ns    = map (\ye -> if !(f32.isnan ye) then 1 else 0) yh
                        |> reduce (+) 0
            let sigma = map (\i -> if i < ns then y_error[i] else 0.0) (iota n)
                        |> map (\ a -> a*a ) |> reduce (+) 0.0
            let sigma = f32.sqrt ( sigma / (r32 (ns-k2p2)) )
            let h     = t32 ( (r32 ns) * hfrac )
            in  (h, ns, sigma)
         ) Yh y_errors

  ---------------------------------------------
  -- 7. moving sums first and bounds:        --
  ---------------------------------------------
  let hmax = reduce_comm (i32.max) 0 hs
  let MO_fsts = zip3 y_errors nss hs |>
    map (\(y_error, ns, h) -> 
            map (\i -> if i < h then y_error[i + ns-h+1] else 0.0) (iota hmax)
            |> reduce (+) 0.0 
        ) |> intrinsics.opaque

  let BOUND = map (\q -> let t   = n+1+q
                         let time = mappingindices[t-1]
                         let tmp = logplus ((r32 time) / (r32 mappingindices[N-1]))
                         in  lam * (f32.sqrt tmp)
                  ) (iota (N-n))

  ---------------------------------------------
  -- 8. moving sums computation:             --
  ---------------------------------------------
  let (breaks, means) = zip (zip4 Nss nss sigmas hs) (zip3 MO_fsts y_errors val_indss) |>
    map (\ ( (Ns,ns,sigma, h), (MO_fst,y_error,val_inds) ) ->
            let MO = map (\j -> if j >= Ns-ns then 0.0
                                else if j == 0 then MO_fst
                                else (-y_error[ns-h+j] + y_error[ns+j])
                         ) (indices BOUND) |> scan (+) 0.0
	    
            let MO' = map (\mo -> mo / (sigma * (f32.sqrt (r32 ns))) ) MO
	        let (is_break, fst_break) = 
		    map3 (\mo' b j ->  if j < Ns - ns && !(f32.isnan mo')
				      then ( (f32.abs mo') > b, j )
				      else ( false, j )
		         ) MO' BOUND (indices BOUND)
		        |> reduce (\ (b1,i1) (b2,i2) -> 
                                if b1 then (b1,i1) 
                                else if b2 then (b2, i2)
                                else (b1,i1) 
              	      	     ) (false, -1)
	        let mean = map2 (\x j -> if j < Ns - ns then x else 0.0 ) MO' (indices BOUND)
			    |> reduce (+) 0.0

	        let fst_break' = if !is_break then -1
                             else let adj_break = adjustValInds n ns Ns val_inds fst_break
                                  in  adj_break
            let fst_break' = if ns <=5 || Ns-ns <= 5 then -2 else fst_break'

            in (fst_break', mean)
        ) |> unzip

  in (breaks, means)
