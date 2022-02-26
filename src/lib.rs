const MULTITHREADING_INPUT_LENGTH_THRESHOLD: usize = 128;

fn should_multithread(length: usize) -> bool {
    length >= MULTITHREADING_INPUT_LENGTH_THRESHOLD
}

// Why have I not managed to find such a function in the std library? Am I missing something?
// TODO should probably return an iterator instead of a vector?
/// When `n` doesn't divide `vec.len()`, parts differ in size by at most 1. Big parts go first in the returned `Vec`.
///
/// # Panics
///
/// If `n <= 0`. TODO how about making it `NonZeroUsize` instead?
fn split_evenly_into_n_parts<T>(vec: Vec<T>, n: usize) -> Vec<Vec<T>> {
    assert!(n >= 1);

    let mut parts = Vec::with_capacity(n);

    let small_part_size = vec.len() / n;
    let big_part_size = small_part_size + 1;
    let big_part_count = vec.len() % n;

    // let mut split_start = 0;
    let mut part_size = big_part_size;

    let mut vec = vec;
    for part_i in 0..n {
        if part_i >= big_part_count {
            part_size = small_part_size;
        }
        // let split_end = split_start + part_size;

        // TODO need to avoid moving stuff in memory.
        parts.push(vec.drain(0..part_size).collect());

        // split_start = split_end;
    }

    parts
}

/// The returned Vec is of the same length as `inputs`, with unchanged order.
pub fn process_vec<T: Send, R: Send>(
    inputs: Vec<T>,
    process_single: fn(t: T) -> R
) -> Vec<R> {
    if !should_multithread(inputs.len()) {
        return inputs.into_iter().map(process_single).collect();
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

        let inputs_parts = split_evenly_into_n_parts(inputs, thread_count);

        // let mut results_parts = Vec::with_capacity(thread_count);
        let mut results_parts: Vec<Vec<R>> = Vec::with_capacity(thread_count);
        // let mut results_parts = Vec::with_capacity::<Vec<Vec<R>>>(thread_count);

        crossbeam_utils::thread::scope(|scope| {
            let mut handles = Vec::with_capacity(thread_count);
            for inputs_part in inputs_parts {
                let handle = scope.spawn(move |_| {
                    inputs_part.into_iter().map(process_single).collect()

                    // let mut results_part = Vec::with_capacity(inputs_part.len());
                    // for input in inputs_part.into_iter() {
                    //     results_part.push(process_single(input));
                    // }
                    // results_part
                });
                handles.push(handle);
            }
            // drop(input);
            for handle in handles {
                results_parts.push(handle.join().unwrap());
            }
        }).unwrap();
        // drop(inputs_parts);

        // TODO is `flatten` efficient here?
        results_parts.into_iter().flatten().collect()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    use rand::{Rng, SeedableRng};

    fn test_random_numbers(length: usize, process_single: fn(t: i32) -> i32) {
        let mut inputs = Vec::with_capacity(length);
        let mut rng = rand::rngs::SmallRng::seed_from_u64(42);

        for _ in 0..length {
            // The distribution range is fundamentally arbitrary, but is chosen so that if you want
            // to test by squaring the number, you don't get an overflow.
            inputs.push(rng.gen_range(-i32::pow(2, 15)..i32::pow(2, 15)));
        }
        assert_eq!(inputs.len(), length);

        assert_eq!(
            inputs.iter().map(|&x| process_single(x)).collect::<Vec<_>>(),
            process_vec(inputs, process_single),
        )
    }

    // TODO separate tests for this function?
    #[test]
    fn split_evenly_into_n_parts_simple() {
        let vec = vec![1,2,3,4,5];
        let split = split_evenly_into_n_parts(vec, 3);
        assert_eq!(
            vec![vec![1, 2], vec![3, 4], vec![5]],
            split,
        )
    }
    #[test]
    fn split_evenly_into_n_parts_parts_more_than_length() {
        let vec = vec![1,2,3];
        let split = split_evenly_into_n_parts(vec, 5);
        assert_eq!(
            // vec![vec![], vec![1], vec![], vec![2], vec![3]],
            vec![vec![1], vec![2], vec![3], vec![], vec![]],
            split,
        )
    }
    #[test]
    #[should_panic]
    fn split_evenly_into_n_parts_panic_if_less_than_1_parts() {
        split_evenly_into_n_parts(vec![1,2,3], 0);
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
            let mut vec = Vec::with_capacity(input_len);
            for i in 0..input_len {
                vec.push(i);
            }
            let parts = split_evenly_into_n_parts(vec, num_parts);

            // Check if the parts are of (almost) the same size.
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

            // Check if the order of the elements is preserved.
            let mut expected_next = 0;
            for part in parts {
                for el in part {
                    assert_eq!(el, expected_next);
                    expected_next += 1;
                }
            }
        }
    }

    // TODO to make sure both the single-threaded and the multithreaded branches are tested, perhaps it would
    // be better to look at code coverage?
    #[test]
    fn numbers_single_thread() {
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
    }
}
