use std::fs::File;
use std::io::{BufWriter, Write};
use std::ops::{Add, AddAssign, Sub, SubAssign};

use rand::{thread_rng, Rng};

const M: usize = 200;
const SAMPLE: usize = M * M;
const RELAXATION: usize = 200;
const INTERVAL: usize = 10;
const TOL: f64 = 1e-1;
const EPSILONX: f64 = 5.0 * TOL;
const EPSILONY: f64 = EPSILONX;
const L: usize = 10;

const ITERATION: usize = 2000;
const RATE: f64 = 0.01;
const STARTING_POINT: Point = Point { x: 1.0, y: 1.0 };

#[derive(Clone, Copy, Debug)]
struct Point {
    x: f64,
    y: f64,
}

impl Point {
    fn norm(&self) -> f64 {
        self.x.hypot(self.y)
    }

    fn norm2(&self) -> f64 {
        self.x * self.x + self.y * self.y
    }

    fn scale(&self, rate: f64) -> Self {
        Self {
            x: self.x * rate,
            y: self.y * rate,
        }
    }
}

impl Add for Point {
    type Output = Self;

    fn add(self, other: Self) -> Self {
        Self {
            x: self.x + other.x,
            y: self.y + other.y,
        }
    }
}

impl AddAssign for Point {
    fn add_assign(&mut self, other: Self) {
        *self = Self {
            x: self.x + other.x,
            y: self.y + other.y,
        };
    }
}

impl Sub for Point {
    type Output = Self;

    fn sub(self, other: Self) -> Self {
        Self {
            x: self.x - other.x,
            y: self.y - other.y,
        }
    }
}

impl SubAssign for Point {
    fn sub_assign(&mut self, other: Self) {
        *self = Self {
            x: self.x - other.x,
            y: self.y - other.y,
        };
    }
}

#[derive(Clone, Copy, Debug)]
struct ChialvoMap {
    a: f64,
    b: f64,
    c: f64,
    k: f64,
}

impl ChialvoMap {
    fn forward(&self, r: Point) -> Point {
        Point {
            x: r.x * r.x * (r.y - r.x).exp() + self.k,
            y: self.a * r.y - self.b * r.x + self.c,
        }
    }

    fn backward(&self, r: Point, tol: f64) -> Point {
        let mut p = STARTING_POINT;
        for _ in 0..ITERATION {
            let q = self.forward(p);
            if (r - q).norm2() < tol {
                break;
            }
            let f = Point {
                x: (2.0 - p.x) * p.x * (p.y - p.x).exp(),
                y: p.x * p.x * (p.y - p.x).exp(),
            };
            let gradient = Point {
                x: 2.0 * f.x * (q.x - r.x) - 2.0 * self.b * (q.y - r.y),
                y: 2.0 * f.y * (q.x - r.x) + 2.0 * self.a * (q.y - r.y),
            };
            p -= gradient.scale(RATE);
        }
        p
    }

    fn sequence(&self, mut r: Point, n: usize) -> Vec<Point> {
        let mut seq = Vec::with_capacity(n);
        seq.push(r);
        for _ in 1..n {
            r = self.forward(r);
            seq.push(r);
        }
        seq
    }
}

#[derive(Clone, Copy, Debug)]
struct Agent {
    r: Point,
    w: f64,
}

type Population = Vec<Vec<Agent>>;

fn population<R: Rng>(seq: &[Point], rng: &mut R) -> Population {
    seq.iter()
        .map(|&rs| {
            (0..M)
                .map(|_| {
                    let r = Point {
                        x: rng.gen_range(0.0..=2.0),
                        y: rng.gen_range(0.0..=2.0),
                    };
                    let w = (r - rs).norm2();
                    Agent { r, w }
                })
                .collect()
        })
        .collect()
}

fn population_dynamics<R: Rng>(
    beta: f64,
    mu: &mut Population,
    nu: &mut Population,
    seq: &[Point],
    map: ChialvoMap,
    rng: &mut R,
) {
    for (n, &rs) in seq.iter().enumerate().skip(1) {
        for _ in 0..M {
            let (i, j) = (rng.gen_range(0..M), rng.gen_range(0..M));
            let p = mu[n][i];
            let q = mu[n - 1][j];
            let r = map.forward(q.r);
            let w = (r - rs).norm2();
            if (-beta * (w - p.w)).exp() > rng.gen() {
                mu[n][i] = Agent { r, w };
            }
        }
    }
    for (n, &rs) in seq.iter().enumerate().rev().skip(1) {
        for _ in 0..M {
            let (i, j) = (rng.gen_range(0..M), rng.gen_range(0..M));
            let p = nu[n][i];
            let q = nu[n + 1][j];
            let r = map.backward(q.r, TOL);
            let w = (r - rs).norm2();
            if (-beta * (w - p.w)).exp() > rng.gen() {
                nu[n][i] = Agent { r, w };
            }
        }
    }
}

fn grid(p: Point) -> Vec<Point> {
    let xmin = (p.x - EPSILONX).max(0.0);
    let xmax = p.x + EPSILONX;
    let ymin = p.y - EPSILONY;
    let ymax = p.y + EPSILONY;
    let xstep = (xmax - xmin) / L as f64;
    let ystep = (ymax - ymin) / L as f64;
    let mut points = Vec::with_capacity((L + 1) * (L + 1));
    let ys = (0..=L).map(|y| y as f64 * ystep + ymin).collect::<Vec<_>>();
    for x in (0..=L).map(|x| x as f64 * xstep + xmin) {
        for &y in &ys {
            points.push(Point { x, y });
        }
    }
    points
}

fn z_node<R: Rng>(
    n: usize,
    rs: Point,
    beta: f64,
    mu: &Population,
    nu: &Population,
    map: ChialvoMap,
    rng: &mut R,
) -> (f64, f64) {
    let mut e = 0.0;
    let mut z = 0.0;
    if n == 0 {
        for _ in 0..SAMPLE {
            let i = rng.gen_range(0..M);
            let p = nu[1][i];
            let q = map.backward(p.r, TOL);
            for r in grid(q) {
                if (p.r - map.forward(r)).norm() < TOL {
                    let err = (r - rs).norm2();
                    let w = (-beta * err).exp();
                    e += w * err;
                    z += w;
                }
            }
        }
    } else if n == mu.len() - 1 {
        for _ in 0..SAMPLE {
            let i = rng.gen_range(0..M);
            let p = mu[n - 1][i];
            let q = map.forward(p.r);
            for r in grid(q) {
                if (p.r - map.backward(r, TOL)).norm() < TOL {
                    let err = (r - rs).norm2();
                    let w = (-beta * err).exp();
                    e += w * err;
                    z += w;
                }
            }
        }
    } else {
        for _ in 0..SAMPLE {
            let (i, j) = (rng.gen_range(0..M), rng.gen_range(0..M));
            let pf = mu[n - 1][i];
            let pb = nu[n + 1][j];
            let q = map.forward(pf.r);
            for r in grid(q) {
                if (pb.r - map.forward(r)).norm() < TOL {
                    let err = (r - rs).norm2();
                    let w = (-beta * err).exp();
                    e += w * err;
                    z += w;
                }
            }
        }
    }
    (e / SAMPLE as f64, z / SAMPLE as f64)
}

fn z_edge<R: Rng>(n: usize, mu: &Population, nu: &Population, map: ChialvoMap, rng: &mut R) -> f64 {
    let mut z = 0.0;
    for _ in 0..SAMPLE {
        let (i, j) = (rng.gen_range(0..M), rng.gen_range(0..M));
        let p = mu[n][i];
        let q = nu[n + 1][j];
        if (q.r - map.forward(p.r)).norm() < TOL {
            z += 1.0;
        }
    }
    z / SAMPLE as f64
}

fn energy_entropy<R: Rng>(
    beta: f64,
    mu: &Population,
    nu: &Population,
    seq: &[Point],
    map: ChialvoMap,
    rng: &mut R,
) -> (f64, f64) {
    let mut energy = 0.0;
    let mut phi = 0.0;
    for (n, &rs) in seq.iter().enumerate() {
        let (e, z) = z_node(n, rs, beta, mu, nu, map, rng);
        energy += e;
        phi += z.ln();
        if n != seq.len() - 1 {
            phi -= z_edge(n, mu, nu, map, rng).ln();
        }
    }
    energy /= seq.len() as f64;
    let entropy = phi / seq.len() as f64 + beta * energy;
    (energy, entropy)
}

fn run(map: ChialvoMap, r: Point, n: usize, beta: f64, phase: &str) {
    let seq = map.sequence(r, n);
    let mut rng = thread_rng();
    let mut mu = population(&seq, &mut rng); // n -> n + 1
    let mut nu = population(&seq, &mut rng); // n -> n - 1
    for _ in 0..RELAXATION {
        population_dynamics(beta, &mut mu, &mut nu, &seq, map, &mut rng);
    }
    let now = std::time::Instant::now();
    let filename = format!("{phase}_{n}_{beta}.csv");
    let file = File::create(filename).unwrap();
    let mut buf = BufWriter::new(file);
    for i in 0.. {
        for _ in 0..INTERVAL {
            population_dynamics(beta, &mut mu, &mut nu, &seq, map, &mut rng);
        }
        let (energy, entropy) = energy_entropy(beta, &mu, &nu, &seq, map, &mut rng);
        if energy.is_finite() && entropy.is_finite() {
            writeln!(buf, "{beta},{i},{energy},{entropy}").unwrap();
            buf.flush().unwrap();
        } else {
            eprintln!("β: {beta} => entropy is NAN or INF!");
        }
        let elapsed = now.elapsed().as_secs();
        let hrs = elapsed / 3600;
        let mins = (elapsed / 60) % 60;
        let secs = elapsed % 60;
        eprintln!("β: {beta} => {i} => {hrs}:{mins:02}:{secs:02}");
    }
}

fn main() {
    let args = std::env::args().collect::<Vec<_>>();
    let n = args[1].parse::<usize>().unwrap();
    let beta = args[2].parse::<f64>().unwrap();
    let chaotic = ChialvoMap {
        a: 0.89,
        b: 0.18,
        c: 0.28,
        k: 0.023,
    };
    let excitable = ChialvoMap {
        a: 0.89,
        b: 0.60,
        c: 0.28,
        k: 0.03,
    };
    let (phase, map) = if args[3].parse::<bool>().unwrap() {
        ("chaotic", chaotic)
    } else {
        ("excitable", excitable)
    };
    let r = Point { x: 1.0, y: 0.3 };
    run(map, r, n, beta, phase);
}
