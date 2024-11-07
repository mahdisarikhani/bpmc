use std::fs::File;
use std::io::{BufWriter, Write};
use std::time::Instant;

use rand::{seq::SliceRandom, thread_rng};
use rayon::iter::{IntoParallelIterator, ParallelIterator};

#[derive(Clone, Copy, Debug)]
struct Point {
    x: f64,
    y: f64,
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

    fn sequence(&self, mut r: Point, n: usize) -> Vec<Point> {
        let mut seq = Vec::with_capacity(n);
        seq.push(r);
        for _ in 1..n {
            r = self.forward(r);
            seq.push(r);
        }
        seq
    }

    fn randomize<R: rand::Rng>(&mut self, eps: f64, rng: &mut R) {
        self.a = (self.a + self.a * rng.gen_range(-eps..=eps)).clamp(0.0, 1.0);
        self.b = (self.b + self.b * rng.gen_range(-eps..=eps)).clamp(0.0, 1.0);
        self.c = (self.c + self.c * rng.gen_range(-eps..=eps)).max(0.0);
        self.k = (self.k + self.k * rng.gen_range(-eps..=eps)).max(0.0);
    }
}

fn error(map: ChialvoMap, seq: &[Point]) -> f64 {
    let error = seq
        .iter()
        .zip(map.sequence(seq[0], seq.len()))
        .map(|(p, q)| (p.x - q.x).powi(2) + (p.y - q.y).powi(2))
        .sum::<f64>();
    (error / seq.len() as f64).sqrt()
}

fn mcstep<R: rand::Rng>(
    mut map: ChialvoMap,
    seq: &[Point],
    step: f64,
    rng: &mut R,
) -> (ChialvoMap, f64) {
    let mut err = error(map, seq);
    let mut params = vec!['a', 'b', 'c', 'k'];
    params.shuffle(rng);
    for p in params {
        let mut tmp = map;
        let rand = step * rng.gen_range(-1.0..=1.0);
        match p {
            'a' => tmp.a = (map.a + rand).clamp(0.0, 1.0),
            'b' => tmp.b = (map.b + rand).clamp(0.0, 1.0),
            'c' => tmp.c = (map.c + rand).max(0.0),
            'k' => tmp.k = (map.k + rand).max(0.0),
            _ => (),
        }
        let e = error(tmp, seq);
        if e < err {
            err = e;
            map = tmp;
        }
    }
    (map, err)
}

fn mc<R: rand::Rng>(mut map: ChialvoMap, n: usize, t: u64, eps: f64, rng: &mut R) -> String {
    let init = map;
    let r = Point {
        x: rng.gen_range(0.0..=2.0),
        y: rng.gen_range(0.0..=2.0),
    };
    let seq = map.sequence(r, n);
    map.randomize(eps, rng);
    let mut error = f64::MAX;
    let mut i = 0;
    while error > 1e-6 && i < t {
        let rate = (t - i) as f64 / t as f64;
        (map, error) = mcstep(map, &seq, rate, rng);
        i += 1;
    }
    format!(
        "{},{},{},{},{},{},{},{},{},{},{},{}",
        init.a, init.b, init.c, init.k, map.a, map.b, map.c, map.k, error, i, n, eps
    )
}

fn sweep(map: ChialvoMap, n: usize, t: u64, ens: u64, eps: f64) -> Vec<String> {
    (0..ens)
        .into_par_iter()
        .map(|_| mc(map, n, t, eps, &mut thread_rng()))
        .collect()
}

fn run(map: ChialvoMap, phase: &str, t: u64, ens: u64) {
    let filename = format!("{phase}_t{}_ens{}.csv", t.ilog2(), ens.ilog2());
    let file = File::create(filename).unwrap();
    let mut buf = BufWriter::new(file);
    writeln!(buf, "a,b,c,k,af,bf,cf,kf,error,t,n,epsilon").unwrap();
    eprintln!("{}", phase);
    for eps in (1..=5).map(|x| x as f64 * 0.1) {
        for n in 2..=32 {
            let now = Instant::now();
            let results = sweep(map, n, t, ens, eps);
            for r in results {
                writeln!(buf, "{r}").unwrap();
            }
            eprintln!("{eps:.1} {n} => {} secs", now.elapsed().as_secs());
        }
    }
}

fn main() {
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
    let args = std::env::args().collect::<Vec<_>>();
    let (phase, map) = if args[1].parse::<bool>().unwrap() {
        ("chaotic", chaotic)
    } else {
        ("excitable", excitable)
    };
    run(map, phase, 1 << 10, 1 << 10);
}
