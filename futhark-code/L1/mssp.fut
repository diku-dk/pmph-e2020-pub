-- Parallel maximum segment sum
-- ==
-- input { [1, -2, 3, 4, -1, 5, -6, 1] }
-- output { 11 }

--
-- compiled input @ mssp-data.in
-- output @ mssp-data.out

-------
-- You may create a random dataset containing integers
-- in [-10 ... 10] with the command:
--      $ futhark dataset --i32-bounds=-10:10 -b -g [64000000]i32 > mssp-data.in
-- After that you may create a reference output, e.g., named mssp-data.out,
--  by running `mssp-seq` on `mssp-data.in`, and after that you may delete the 
--  new line 5, so that both datasets are executed when running 
--  `futhark bench --backend=c mssp-seq.fut`
-------

type int = i32
let max (x:int, y:int) = i32.max x y

let redOp(x: (int,int,int,int))
         (y: (int,int,int,int)): (int,int,int,int) =
  let (mssx, misx, mcsx, tsx) = x
  let (mssy, misy, mcsy, tsy) = y in
  ( max(mssx, max(mssy, mcsx + misy))
  , max(misx, tsx+misy)
  , max(mcsy, mcsx+tsy)
  , tsx + tsy)

let mapOp (x: int): (int,int,int,int) =
  ( max(x,0)
  , max(x,0)
  , max(x,0)
  , x)

let main(xs: []int): int =
  let (x, _, _, _) =
    reduce redOp (0,0,0,0) (map mapOp xs) in
  x
