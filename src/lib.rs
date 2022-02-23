const MULTITHREADING_INPUT_LENGTH_THRESHOLD: usize = 128;

// fn should_multithread<T>(inputs: &Vec<T>) -> bool {
//     inputs.len() > MULTITHREADING_INPUT_LENGTH_THRESHOLD
// }
fn should_multithread(length: usize) -> bool {
    length >= MULTITHREADING_INPUT_LENGTH_THRESHOLD
}

// fn test<T>(vec: Vec<T>) -> Vec<Vec<T>> {
//     // vec![vec.split_at(2).0.to_owned().to_vec()]
//     // vec![vec.split_at(2).0]
//     // vec.split_at(2).to_owned()
//     vec.chunks(2).to_owned().
// }


// https://stackoverflow.com/a/6861195/10406353 more efficient?
// Why have I not managed to find such a function in the std library? Am I missing something?
// TODO should probably return an iterator instead of a vector?
// fn split_evenly_into_n_parts<T>(vec: Vec<T>, n: usize) -> Vec<Vec<T>> {
// fn split_evenly_into_n_parts<T, const N: usize>(vec: Vec<T>, n: N) -> [&[T]; N] {
// fn split_evenly_into_n_parts<T>(vec: Vec<T>, n: usize) -> Vec<[T]> {
// fn split_evenly_into_n_parts<'a, T: 'a>(vec: &'a [T], n: usize) -> Vec<&'a [T]> {
/// When `n` doesn't divide `vec.len()`, parts differ in size by at most 1. Big parts go first in the returned `Vec`.
///
/// # Panics
///
/// If `n <= 0`. TODO how about making it `NonZeroUsize` instead?
fn split_evenly_into_n_parts<'a, T: 'a>(vec: &'a [T], n: usize) -> Vec<&'a [T]> {
// fn split_evenly_into_n_parts<'a, T: 'a>(vec: &'a [T], n: NonZeroUsize) -> Vec<&'a [T]> {
// fn split_evenly_into_n_parts<'a, T: 'a>(vec: &'a Vec<T>, n: usize) -> Vec<&'a Vec<T>> {
    // vec.chunks(2);
    // (0..n)
    // let split_step_unrounded = vec.len() / n;

    // let (asd1, asd2) = vec![1,2,3].split_at(1);
    // let dsa = asd1.to_owned();

    // let (dfg1 , dfg2) = vec.split_at(2);
    // let fghnjknjkghf = dfg1.to_vec();
    // let fghghghf = vec![fghnjknjkghf];
    // return fghghghf;

    // let parts = Vec::<&[T]>::with_capacity(n);

    // let n = n.get();

    assert!(n >= 1);

    let mut parts = Vec::with_capacity(n);

    // let mut split_start = 0;
    // // let mut tail = vec.as_slice().to_owned();
    // // let mut remainder = vec.as_slice();
    // let mut remainder = vec;
    // // let part;
    // for i in 1..=n {
    //     let split_end = i * vec.len() / n;
    //     dbg!(split_end, remainder.len());
    //     // let (part, new_remainder) = remainder.split_at(split_end);
    //     // remainder = new_remainder;

    //     parts.push(&vec[split_start..split_end]);

    //     // remainder = new_remainder.to_owned();
    //     split_start = split_end;
    //     // parts.push(part.to_owned());
    //     // parts.push(part);
    //     // parts.push(part.to_vec());
    //     // parts.push(part);
    // }
    // // parts.push(remainder);

    let small_part_size = vec.len() / n;
    let big_part_size = small_part_size + 1;
    let big_part_count = vec.len() % n;

    // for i in 0..big_part_count {
    //     // let split_start = big_part_size * i;
    //     parts.push(&vec[big_part_size])
    // }

    let mut split_start = 0;
    let mut part_size = big_part_size;
    // for _ in 0..big_part_count {
    for i in 0..n {
        if i >= big_part_count {
            part_size = small_part_size;
        }
        let split_end = split_start + part_size;
        parts.push(&vec[split_start..split_end]);
        split_start = split_end;
    }

    parts
    // return vec.chunks(n);
}

/// The returned Vec is of the same length as `inputs`, with unchanged order.
// pub fn process_vec<T: std::marker::Send, R: std::fmt::Debug>(
//     inputs: Vec<T>,
//     process_single: fn(t: T) -> R
// ) -> Vec<R> {
// pub fn process_vec<T: std::fmt::Debug, R: std::fmt::Debug>(
//     inputs: Vec<T>,
//     process_single: fn(t: T) -> R
// ) {
// pub fn process_vec<R: std::fmt::Debug>(
//     inputs: Vec<i32>,
//     process_single: fn(t: i32) -> R
// ) {
// pub fn process_vec<'a, R: 'a + std::fmt::Debug>(
pub fn process_vec(
    inputs: Vec<i32>,
    process_single: fn(t: i32) -> i32
) -> Vec<i32> {
    // let mut results = Vec::with_capacity(inputs.len());
    // for input in inputs {
    //     results.push(process_single(input));
    // }
    // results

    // inputs.iter().map(process_single)

    // inputs.iter().map(process_single).collect()

    // if should_multithread(&inputs) {
    if !should_multithread(inputs.len()) {
        return inputs.into_iter().map(process_single).collect();
        // return inputs.iter().map(process_single).collect();
    } else {
        // return inputs.into_iter().map(process_single).collect();
        // return inputs.iter().map(process_single).collect();

        // TODO if this is 1, also don't spawn threads? What about tests though? they'll not
        // cover the multithreaded branch on single-core machines.
        //
        // TODO consider replacing with `std::thread::available_parallelism` in Rust 1.59, after stabilisation.
        // let default_expected_available_paralellism = 8; // TODO not sure if it's right to do this.
        // let available_parallelism =
        //     std::thread::available_parallelism().unwrap_or(default_expected_available_paralellism);
        //
        let cpus = num_cpus::get();
        // let cpus_usize = num_cpus::get();
        // let cpus;
        // unsafe {
        //     cpus = NonZeroUsize::new_unchecked(cpus_usize);
        // }

        // TODO the below code is probably not very smart, because I'm a bit of a noob when it comes to parallelism,
        // so chances are it needs to be rewritten.
        //
        // Need this multiplier in case e.g. a single core is very busy with another process. In that case all
        // threads of our process but the one assigned to that core would finish faster than that one, and will
        // just do nothing until it finishes. If we create more threads, each of them will have a smaller job
        // assigned to it. There's a bit of a cost to it though.
        let tasks_per_cpu = 8;
        // const tasks_per_cpu: NonZeroUsize = NonZeroUsize::new(8).unwrap();
        let thread_count = cpus * tasks_per_cpu;
        // TODO what if thread_count is bigger than `inputs.len()`?
        // let mut handles = Vec::with_capacity(thread_count);

        // let max_chunk_size = (inputs.len() / thread_count).ceil();
        // let input_chunks = inputs.chunks
        // for _ in 0..thread_count {
        //     handles.push(std::thread::spawn())
        // }

        let inputs_parts = split_evenly_into_n_parts(&inputs, thread_count);
        // let inputs_parts = inputs.chunks(2);
        // let inputs_parts = [inputs];

        // dbg!(inputs);
        // println!("{}", inputs);

        // let mut results_parts = Vec::<Arc<Vec<R>>>::with_capacity(thread_count);

        let mut results_parts = Vec::with_capacity(thread_count);
        crossbeam_utils::thread::scope(|scope| {
            let mut handles = Vec::with_capacity(thread_count);
            for inputs_part in inputs_parts {
                let handle = scope.spawn(|_| {
                    let mut results_part = Vec::with_capacity(inputs_part.len());

                    // part.into_iter().map(process_single).collect()
                    // part.into_iter().map(|&x| process_single(x)).collect()
                    // for input in inputs_part {
                    for input in inputs_part.to_owned() {
                        results_part.push(process_single(input));
                        // results_part.push(process_single(*input));
                        // results_part.push(1);
                    }
                    results_part
                });
                handles.push(handle);
            }
            for handle in handles {
                results_parts.push(handle.join().unwrap());
            }
        }).unwrap();

        // for inputs_part in inputs_parts {
            // results_parts.push(Arc::new(Vec::with_capacity(part.len())));
            // let results_part = Arc::new(Vec::with_capacity(inputs_part.len()));
            // let results_part = Arc::new(Vec::with_capacity(inputs_part.len()));
            // results_parts.push(results_part);

            // let closure = |a: Vec<T>| {
            //     // let mut results_part = Vec::with_capacity(inputs_part.len());
            //     let mut results_part = Vec::with_capacity(a.len());

            //     // part.into_iter().map(process_single).collect()
            //     // part.into_iter().map(|&x| process_single(x)).collect()
            //     // for input in inputs_part {
            //     for input in a {
            //         // results_part.push(process_single(input));
            //         results_part.push(1);
            //     }

            //     // dbg!(results_part.len());
            //     dbg!(results_part);
            //     // results_part
            // };
            // std::thread::spawn(move || { closure(inputs) });

            // let handle = std::thread::spawn(move || {
            //     // let mut results_part = Vec::with_capacity(inputs_part.len());
            //     let mut results_part = Vec::with_capacity(inputs.len());

            //     // part.into_iter().map(process_single).collect()
            //     // part.into_iter().map(|&x| process_single(x)).collect()
            //     // for input in inputs_part {
            //     for input in inputs {
            //         // results_part.push(process_single(input));
            //         results_part.push(process_single(input));
            //     }

            //     // dbg!(results_part.len());
            //     dbg!(results_part);
            //     // results_part
            // });
            // handles.push(handle);

            // let inputs = Arc::new(inputs);
            // let handle = std::thread::spawn(move || {
            //     let inputs = Arc::clone(&inputs);
            //     // let mut results_part = Vec::with_capacity(inputs_part.len());
            //     let mut results_part = Vec::with_capacity(inputs.len());

            //     // part.into_iter().map(process_single).collect()
            //     // part.into_iter().map(|&x| process_single(x)).collect()
            //     // for input in inputs_part {
            //     // for input in inputs.into_iter() {
            //     for input in inputs.iter() {
            //         // results_part.push(process_single(input));
            //         results_part.push(process_single(*input));
            //     }

            //     // dbg!(results_part.len());
            //     dbg!(results_part);
            //     // results_part
            // });
            // handles.push(handle);

        // }

        // let results_parts: Vec<&[R]>
        // let results_parts = Vec::<&[R]>::with_capacity(thread_count);
        // let results_parts = Vec::<Vec<R>>::with_capacity(thread_count);
        // for handle in handles {
        //     // let result = handle.join().unwrap();
        //     let results_part = handle.join().unwrap();
        //     results_parts.push(result);
        // }

        // let results = Vec::with_capacity(inputs.len());
        // for i in 0..thread_count {
        //     let results_part = handles[i].join().unwrap();
        //     // handles[i].join().unwrap();
        //     // results.concat(results_part);
        //     // for result in results_part.into_iter() {
        //     //     results.push(result);
        //     // }
        // }

        // for results_part in results_parts {
        //     for result in results_part.into_iter() {
        //         results.push(result);
        //     }
        // }
        // results
        // return results_parts.concat();


        // return Vec::from(results);
        results_parts.into_iter().flatten().collect()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    use rand::{Rng, SeedableRng};
    // use rand::rngs::SmallRng;

    // fn test_random_numbers<R: std::fmt::Debug + std::cmp::PartialEq>(length: usize, process_single: fn(t: i32) -> i32) {
    fn test_random_numbers(length: usize, process_single: fn(t: i32) -> i32) {
    // fn test_square_random_numbers(length: usize) {
        let mut inputs = Vec::with_capacity(length);
        let mut rng = rand::rngs::SmallRng::seed_from_u64(42);
        // inputs.fill(rng.gen::<i32>());
        // inputs.fill_with(|| rng.gen::<i32>());

        // inputs.fill_with(|| rng.gen::<i32>());
        for _ in 0..length {
            // The distribution range is fundamentally arbitrary, but is chosen so that if you want
            // to test by squaring the number, you don't get an overflow.
            inputs.push(rng.gen_range(-i32::pow(2, 15)..i32::pow(2, 15)));
        }
        assert_eq!(inputs.len(), length);

        assert_eq!(
            // inputs.iter().map(|x| x * x).collect::<Vec<_>>(),
            // process_vec(inputs, |x| x * x),
            inputs.iter().map(|&x| process_single(x)).collect::<Vec<_>>(),
            // process_vec(inputs, process_single),
            process_vec(inputs, process_single),
        )
    }

    // TODO separate tests for this function?
    #[test]
    fn split_evenly_into_n_parts_simple() {
        // let vec = Vec::from([0; 4]);
        let vec = vec![1,2,3,4,5];
        let split = split_evenly_into_n_parts(&vec, 3);
        // assert_eq!(split.len)
        assert_eq!(
            // vec![vec![1], vec![2, 3], vec![4, 5]],
            vec![vec![1, 2], vec![3, 4], vec![5]],
            split,
        )
    }
    #[test]
    fn split_evenly_into_n_parts_parts_more_than_length() {
        // let vec = Vec::from([0; 4]);
        let vec = vec![1,2,3];
        let split = split_evenly_into_n_parts(&vec, 5);
        assert_eq!(
            // vec![vec![], vec![1], vec![], vec![2], vec![3]],
            vec![vec![1], vec![2], vec![3], vec![], vec![]],
            split,
        )
    }
    #[test]
    #[should_panic]
    fn split_evenly_into_n_parts_panic_if_less_than_1_parts() {
        // let vec = Vec::from([0; 4]);
        // let vec = vec![1,2,3];
        split_evenly_into_n_parts(&vec![1,2,3], 0);
    }
    #[test]
    fn split_evenly_into_n_parts_different_lengths() {
        let combos = [
            (18, 4),
            (16, 4),
            (91, 8),
            (0, 19),
            (651981, 573),
        ];
        for (input_len, num_parts) in combos {
            // let vec = Vec::from(0..input_len);
            let mut vec = Vec::with_capacity(input_len);
            for i in 0..input_len {
                vec.push(i);
            }
            let parts = split_evenly_into_n_parts(&vec, num_parts);
            // assert!(parts.windows(2).all(|(a, b)| { a.len() - b.len() }));

            let min_part_len = input_len / num_parts;
            let max_part_len = if input_len % num_parts > 0 {
                min_part_len + 1
            } else {
                min_part_len
            };
            assert!(parts.iter().all(|p| {
                p.len() == min_part_len
                || p.len() == max_part_len
            }));

            let parts_len_sum: usize = parts.iter().map(|p| p.len()).sum();
            assert_eq!(parts_len_sum, input_len);

            // let order_preserved = parts.windows(2).all(|w| {
            //     // p1.last() == p2.first() + 1
            //     w[0].last() == w[1].first() + 1
            //     let last = w[0].last();
            //     let first = w[1].first();
            //     // match last {

            //     // }
            // });

            // Check if the order of the elements is preserved.
            let mut expected_next = 0;
            for part in parts {
                for el in part {
                    assert_eq!(*el, expected_next);
                    expected_next += 1;
                }
            }
        }
    }

    // TODO to make sure both the single-threaded and the multithreaded branches are tested, perhaps it would
    // be better to look at code coverage?
    #[test]
    fn numbers_single_thread() {
        // let inputs = vec![1, 3, 3, 7, 12, -11];
        // assert!(!should_multithread(&inputs));
        // let expected = vec![1, 9, 9, 49, 144, 121];
        // let results = process_vec(inputs, |x| x * x);
    
        // // assert_eq!(results.len(), expected.len());
    
        // // for (i, &e) in expected.iter().enumerate() {
        // //     assert_eq!(results[i], e);
        // // }
        // assert_eq!(results, expected);

        let inputs = vec![1, 3, 3, 7, 12, -11];
        let expected = vec![1, 9, 9, 49, 144, 121];
        assert!(!should_multithread(inputs.len()));
        assert_eq!(process_vec(inputs, |x| x * x), expected);
    }
    #[test]
    fn numbers_multithread() {
        let length = MULTITHREADING_INPUT_LENGTH_THRESHOLD * 42 + 1337;
        assert!(should_multithread(length));
        test_random_numbers(length, |x| 2 * x);
        // test_random_numbers(length, |x| x / 2);
    }
}
