use rand::{SeedableRng, RngCore};
use rand_pcg::Lcg128Xsl64;
use std::f64::consts::PI;
use rayon::prelude::*;
use gravity::state::*;
use std::io::{BufWriter, Write};
use std::fs::File;

const NUM_THREADS: usize = 6;
const NUM_OF_ORBIT: usize = 98_304;
const SAMPLES_PER_FILE: usize = 100;
const POSITION_OFFSET: f64 = 0.5;
const VELOCITY_OFFSET: f64 = 0.7;
const SAMPLE_STEP: f64 = 0.4;
const MASS: f64 = 1.0;
const U64_MAX: f64 = std::u64::MAX as f64;

fn comp_delta_t(acc: &Acceleration, jerk: &Jerk) -> f64 {
    0.001 * ((acc.x * acc.x + acc.y * acc.y) / (jerk.x * jerk.x + jerk.y * jerk.y)).sqrt()
}

fn one_step(mass: f64, pos: &mut Position, vel: &mut Velocity) -> u64 {
    let mut time = 0.0;
    let mut loop_times = 0;
    loop {
        loop_times += 1;
        let acc0 = Acceleration::new(mass, &pos);
        let jerk0 = Jerk::new(mass, &pos, &vel);
        let delta_t = comp_delta_t(&acc0, &jerk0);
        if delta_t > SAMPLE_STEP - time {
            let delta_t = SAMPLE_STEP - time;
            pos.predict(&vel, &acc0, &jerk0, delta_t);
            vel.predict(&acc0, &jerk0, delta_t);

            let acc1 = Acceleration::new(mass, &pos);
            let jerk1 = Jerk::new(mass, &pos, &vel);
            let (coe0, coe1) = Coefficient::new(&acc0, &acc1, &jerk0, &jerk1, delta_t);
            pos.correct(&coe0, &coe1, delta_t);
            vel.correct(&coe0, &coe1, delta_t);
            break
        }

        pos.predict(&vel, &acc0, &jerk0, delta_t);
        vel.predict(&acc0, &jerk0, delta_t);

        let acc1 = Acceleration::new(mass, &pos);
        let jerk1 = Jerk::new(mass, &pos, &vel);
        let (coe0, coe1) = Coefficient::new(&acc0, &acc1, &jerk0, &jerk1, delta_t);
        pos.correct(&coe0, &coe1, delta_t);
        vel.correct(&coe0, &coe1, delta_t);
        time += delta_t;
    }
    loop_times
}

fn initialize(mut rng: Lcg128Xsl64) -> Vec<(Position, Velocity)> {
    let mut initial_state = Vec::with_capacity(NUM_OF_ORBIT);

    while initial_state.len() < NUM_OF_ORBIT {
        let radius_of_position = (rng.next_u64() as f64 / U64_MAX) + POSITION_OFFSET;
        let radius_of_velocity = (rng.next_u64() as f64 / U64_MAX) + VELOCITY_OFFSET;
        let theta_of_position = (rng.next_u64() as f64 / U64_MAX) * 2.0 * PI;
        let theta_of_velocity = (rng.next_u64() as f64 / U64_MAX) * 2.0 * PI;
        if radius_of_position * radius_of_velocity * radius_of_velocity > 2.1 { continue }

        let position = Position {
            x: radius_of_position * theta_of_position.cos(),
            y: radius_of_position * theta_of_position.sin(),
            abs: radius_of_position,
        };
        let velocity = Velocity {
            x: radius_of_velocity * theta_of_velocity.cos(),
            y: radius_of_velocity * theta_of_velocity.sin(),
        };

        initial_state.push((position, velocity));
    }
    initial_state
}

fn main() {
    rayon::ThreadPoolBuilder::new().num_threads(NUM_THREADS).build_global().unwrap();

    for &(seed, data_name) in [(101u64, "train"), (102, "validation"), (104, "test")].iter() {
        let rng = Lcg128Xsl64::seed_from_u64(seed);
        let initial_state = initialize(rng);
        {
            let file_name = format!("./data/{}_initial_state.txt", data_name);
            let mut file_stream = BufWriter::new(File::create(&file_name).unwrap());
            for &(pos, vel) in initial_state.iter() {
                writeln!(&mut file_stream, "{} {} {} {}", pos.x, pos.y, vel.x, vel.y).unwrap();
            }
        }
        initial_state.into_par_iter().enumerate().for_each(|(ind, (mut pos, mut vel))| {
            let file_name = format!("./data/{}/{}.txt", data_name, ind);
            let mut file_stream = BufWriter::new(File::create(&file_name).unwrap());
            writeln!(&mut file_stream, "{} {} {} {} 0", pos.x, pos.y, vel.x, vel.y).unwrap();

            for _ in 0..SAMPLES_PER_FILE {
                let loop_times = one_step(MASS, &mut pos, &mut vel);
                writeln!(&mut file_stream, "{} {} {} {} {}", pos.x, pos.y, vel.x, vel.y, loop_times).unwrap();
            }
        });
    }
}
