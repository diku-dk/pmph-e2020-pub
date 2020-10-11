-- Primes: Naive Version
-- ==
-- compiled input { 30i32 } output { [2,3,5,7,11,13,17,19,23,29] }
-- compiled input { 10000000i32 } auto output

let primesHelp [np1] (sq : i32) (a : *[np1]i32) : [np1]i32 =
  let n = np1 - 1 in
  loop(a) for i < (sq-1) do 
        let i    = i + 2 in
        if a[i] == 0i32 then a
        else
          let m    = (n / i) - 1
          let inds = map (\k -> (k+2)*i) (iota m)
          let vals = replicate m 0
          let a'  = scatter a inds vals
          in  a'

-- Run with $echo "1000000" | ./primes-naive -t /dev/stderr -r 10 > /dev/null
let main (n : i32) : []i32 =
  let np1 = n+1
  let a = map (\i -> if i==0 || i==1 then 0 else 1) (iota np1)
  let sq= t32 (f32.sqrt (r32 n))
  let fl= primesHelp sq a
  in  filter (\i -> fl[i]!=0) (iota np1)

