// // fn test<T: std::fmt::Debug/*  + Send */ /* + 'static */>(v: Vec<T>) {
// //     // let v = vec![1, 2, 3];

// //     let handle = std::thread::spawn(move || {
// //         println!("Here's a vector: {:?}", v);
// //     });

// //     // drop(v); // oh no!

// //     handle.join().unwrap();
// // }

// // // fn test2(vec: Vec<i32>) {
// // fn test2() {
// //     let asd = vec![1,2,3];
// //     // So if we pass a vector like this to a thread, it may outlive the main thread, which would make the &vec
// //     // reference invalid, is this the reason?
// //     // Looks like yes, see the first paragraph https://stackoverflow.com/a/32751956/10406353
// //     let vec = vec![&asd, /* vec![1,2], vec![1,2] */];

// //     test(vec);
// //     // drop(vec);
// // }

// // fn main() {
// //     // test(vec![1,2,3]);
// //     test2();
// // }


// use std::sync::{Arc, Mutex};
// use std::thread;

// fn test<T>(data: Vec<T>, process_single: fn(t: &T) -> i32) {
// // fn test<T>(data: T, process_single: fn(t: &T) -> i32) {
//     let counter = Arc::new(Mutex::new(0));
//     let mut handles = vec![];

//     for i in 0..10 {
//         let counter = Arc::clone(&counter);
//         let handle = thread::spawn(move || {
//             let mut num = counter.lock().unwrap();

//             *num += 1;
//             // *num += process_single(data[i]);
//             // *num += process_single(&data);
//         });
//         handles.push(handle);
//     }

//     for handle in handles {
//         handle.join().unwrap();
//     }

//     println!("Result: {}", *counter.lock().unwrap());
// }

// fn main() {
//     test(vec![1,2,3,4,5,6,7,8,9,10], |x| { x * 2 });
// }
