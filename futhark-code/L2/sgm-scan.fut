-- Segmented scan
--
-- ==
-- input { [2i32,0i32,2i32,0i32]
--         [1i32,2i32,3i32,4i32]
-- }
-- output {
--   [1i32,3i32,3i32,7i32] 
-- }
--
-- compiled random input { [16777216]f32 } auto output

-- run with: $ futhark dataset --i32-bounds=-100:100 -g [8388608]i32 | ./sgm-scan -r 2 -t /dev/stderr > /dev/null

let segmented_scan [n] 't (op: t -> t -> t) (ne: t)
                          (flags: [n]bool) (arr: [n]t) : [n]t =
  let (_, res) = unzip <|
    scan (\(x_flag,x) (y_flag,y) ->
             let fl = x_flag || y_flag
             let vl = if y_flag then y else op x y
             in  (fl, vl)
         ) (false, ne) (zip flags arr)
  in  res

let main [n] (data: [n]i32) : [n]i32 =
  let flags = map (\i -> if (i % 32) == 0 then true else false) (iota n)
  in  segmented_scan (+) 0i32 flags data
