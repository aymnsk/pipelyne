// src/lib.rs - FINAL POLISHED VERSION
use std::fs::File;
use std::io::Write;
use std::fmt;
use rand::Rng;
use ordered_float::OrderedFloat;

// Type aliases for cleaner code
type PipelineStepType = Box<dyn PipelineStep<Input = Vec<f64>, Output = Vec<f64>, Error = String>>;
type SplitResult = (Vec<(Vec<f64>, f64)>, Vec<(Vec<f64>, f64)>);

/// Core Pipeline trait - all steps must implement this
pub trait PipelineStep: fmt::Debug {
    type Input;
    type Output;
    type Error;
    fn process(&mut self, input: Self::Input) -> Result<Self::Output, Self::Error>;
}

/// Pipeline container for chaining steps
#[derive(Debug, Default)]
pub struct Pipeline {
    steps: Vec<PipelineStepType>,
}

impl Pipeline {
    pub fn new() -> Self {
        Self::default()
    }
    
    pub fn add_step(&mut self, step: PipelineStepType) {
        self.steps.push(step);
    }
    
    pub fn run(&mut self, initial_input: Vec<f64>) -> Result<Vec<f64>, String> {
        let mut current_data = initial_input;
        for step in &mut self.steps {
            current_data = step.process(current_data)?;
        }
        Ok(current_data)
    }
    
    pub fn save_info(&self, filename: &str) -> Result<(), String> {
        let info = PipelineInfo {
            num_steps: self.steps.len(),
            step_types: self.steps.iter()
                .map(|step| format!("{:?}", step))
                .collect(),
        };
        
        let json = serde_json::to_string_pretty(&info)
            .map_err(|e| format!("Failed to serialize: {}", e))?;
        
        let mut file = File::create(filename)
            .map_err(|e| format!("Failed to create file: {}", e))?;
        
        file.write_all(json.as_bytes())
            .map_err(|e| format!("Failed to write file: {}", e))?;
        
        println!("💾 Saved pipeline info to {}", filename);
        Ok(())
    }
    
    pub fn load_info(filename: &str) -> Result<PipelineInfo, String> {
        let file = File::open(filename)
            .map_err(|e| format!("Failed to open file: {}", e))?;
        
        let info: PipelineInfo = serde_json::from_reader(file)
            .map_err(|e| format!("Failed to parse JSON: {}", e))?;
        
        println!("📂 Loaded pipeline info: {} steps", info.num_steps);
        Ok(info)
    }
}

#[derive(serde::Serialize, serde::Deserialize, Debug)]
pub struct PipelineInfo {
    pub num_steps: usize,
    pub step_types: Vec<String>,
}

// ===== CSV LOADER =====
#[derive(Debug)]
pub struct CsvLoader {
    filename: String,
    has_headers: bool,
}

impl CsvLoader {
    pub fn new(filename: &str) -> Self {
        CsvLoader {
            filename: filename.to_string(),
            has_headers: true,
        }
    }
    
    pub fn without_headers(filename: &str) -> Self {
        CsvLoader {
            filename: filename.to_string(),
            has_headers: false,
        }
    }
}

impl PipelineStep for CsvLoader {
    type Input = Vec<f64>;
    type Output = Vec<f64>;
    type Error = String;

    fn process(&mut self, _input: Self::Input) -> Result<Self::Output, Self::Error> {
        let file = File::open(&self.filename)
            .map_err(|e| format!("Failed to open file {}: {}", self.filename, e))?;
        
        let mut rdr = csv::ReaderBuilder::new()
            .has_headers(self.has_headers)
            .from_reader(file);
        
        let mut data = Vec::new();
        
        for result in rdr.records() {
            let record = result
                .map_err(|e| format!("Failed to parse CSV: {}", e))?;
            
            if let Some(x_str) = record.get(0) {
                match x_str.parse::<f64>() {
                    Ok(x) => data.push(x),
                    Err(_) => return Err(format!("Failed to parse number: {}", x_str)),
                }
            }
        }
        
        if data.is_empty() {
            return Err("CSV file is empty".to_string());
        }
        
        println!("📊 Loaded {} values from {}", data.len(), self.filename);
        Ok(data)
    }
}

// ===== XY DATA LOADER =====
#[derive(Debug)]
pub struct XyDataLoader {
    filename: String,
    x_col: usize,
    y_col: usize,
}

impl XyDataLoader {
    pub fn new(filename: &str, x_col: usize, y_col: usize) -> Self {
        XyDataLoader {
            filename: filename.to_string(),
            x_col,
            y_col,
        }
    }
}

impl PipelineStep for XyDataLoader {
    type Input = ();
    type Output = (Vec<f64>, Vec<f64>);
    type Error = String;

    fn process(&mut self, _input: Self::Input) -> Result<Self::Output, Self::Error> {
        let file = File::open(&self.filename)
            .map_err(|e| format!("Failed to open file {}: {}", self.filename, e))?;
        
        let mut rdr = csv::ReaderBuilder::new()
            .has_headers(true)
            .from_reader(file);
        
        let mut x_data = Vec::new();
        let mut y_data = Vec::new();
        
        for (i, result) in rdr.records().enumerate() {
            let record = result
                .map_err(|e| format!("Failed to parse CSV at line {}: {}", i+2, e))?;
            
            if let Some(x_str) = record.get(self.x_col) {
                match x_str.parse::<f64>() {
                    Ok(x) => x_data.push(x),
                    Err(_) => return Err(format!("Failed to parse X value '{}' at line {}", 
                                                 x_str, i+2)),
                }
            }
            
            if let Some(y_str) = record.get(self.y_col) {
                match y_str.parse::<f64>() {
                    Ok(y) => y_data.push(y),
                    Err(_) => return Err(format!("Failed to parse y value '{}' at line {}", 
                                                 y_str, i+2)),
                }
            }
        }
        
        if x_data.is_empty() {
            return Err("No valid data found".to_string());
        }
        
        if x_data.len() != y_data.len() {
            return Err(format!("X and y have different lengths: {} vs {}", 
                              x_data.len(), y_data.len()));
        }
        
        println!("📊 Loaded {} X,y pairs from {}", x_data.len(), self.filename);
        Ok((x_data, y_data))
    }
}

// ===== STANDARD SCALER =====
#[derive(Debug, Default)]
pub struct StandardScaler {
    mean: Option<f64>,
    std: Option<f64>,
}

impl StandardScaler {
    pub fn new() -> Self {
        Self::default()
    }
    
    fn compute_mean(&self, data: &[f64]) -> f64 {
        data.iter().sum::<f64>() / data.len() as f64
    }
    
    fn compute_std(&self, data: &[f64], mean: f64) -> f64 {
        let variance: f64 = data.iter()
            .map(|&x| (x - mean).powi(2))
            .sum::<f64>() / data.len() as f64;
        variance.sqrt()
    }
}

impl PipelineStep for StandardScaler {
    type Input = Vec<f64>;
    type Output = Vec<f64>;
    type Error = String;

    fn process(&mut self, input: Self::Input) -> Result<Self::Output, Self::Error> {
        if input.is_empty() {
            return Err("Cannot scale empty data".to_string());
        }
        
        let mean = self.mean.unwrap_or_else(|| self.compute_mean(&input));
        let std = self.std.unwrap_or_else(|| self.compute_std(&input, mean));
        
        if (std - 0.0).abs() < f64::EPSILON {
            return Err("Standard deviation is zero - cannot scale".to_string());
        }
        
        let scaled: Vec<f64> = input.iter()
            .map(|&x| (x - mean) / std)
            .collect();
        
        println!("📐 Scaled data: mean={:.2}, std={:.2}", mean, std);
        Ok(scaled)
    }
}

// ===== TRAIN TEST SPLIT =====
#[derive(Debug)]
pub struct TrainTestSplit {
    test_size: f64,
}

impl TrainTestSplit {
    pub fn new(test_size: f64) -> Self {
        TrainTestSplit { test_size }
    }
}

impl PipelineStep for TrainTestSplit {
    type Input = (Vec<f64>, Vec<f64>);
    type Output = (Vec<f64>, Vec<f64>, Vec<f64>, Vec<f64>);
    type Error = String;

    fn process(&mut self, input: Self::Input) -> Result<Self::Output, Self::Error> {
        let (x, y) = input;
        
        if x.len() != y.len() {
            return Err("X and y must have same length".to_string());
        }
        
        let n = x.len();
        let test_count = (n as f64 * self.test_size).round() as usize;
        
        if test_count == 0 || test_count >= n {
            return Err("Invalid test size".to_string());
        }
        
        let mut indices: Vec<usize> = (0..n).collect();
        indices.reverse();
        
        let test_indices = &indices[0..test_count];
        let train_indices = &indices[test_count..];
        
        let mut x_train = Vec::with_capacity(train_indices.len());
        let mut x_test = Vec::with_capacity(test_indices.len());
        let mut y_train = Vec::with_capacity(train_indices.len());
        let mut y_test = Vec::with_capacity(test_indices.len());
        
        for &idx in train_indices {
            x_train.push(x[idx]);
            y_train.push(y[idx]);
        }
        
        for &idx in test_indices {
            x_test.push(x[idx]);
            y_test.push(y[idx]);
        }
        
        println!("📊 Train/Test Split: {}/{} train, {}/{} test", 
            x_train.len(), n, x_test.len(), n);
        
        Ok((x_train, x_test, y_train, y_test))
    }
}

// ===== LINEAR REGRESSION =====
#[derive(Debug, serde::Serialize, serde::Deserialize, Default)]
pub struct LinearRegression {
    slope: Option<f64>,
    intercept: Option<f64>,
    trained: bool,
}

impl LinearRegression {
    pub fn new() -> Self {
        Self::default()
    }
    
    pub fn train(&mut self, x: &[f64], y: &[f64]) -> Result<(), String> {
        if x.len() != y.len() {
            return Err("X and y must have same length".to_string());
        }
        
        let n = x.len() as f64;
        let x_mean: f64 = x.iter().sum::<f64>() / n;
        let y_mean: f64 = y.iter().sum::<f64>() / n;
        
        let mut numerator = 0.0;
        let mut denominator = 0.0;
        
        for i in 0..x.len() {
            let x_diff = x[i] - x_mean;
            let y_diff = y[i] - y_mean;
            numerator += x_diff * y_diff;
            denominator += x_diff * x_diff;
        }
        
        if denominator.abs() < f64::EPSILON {
            return Err("Cannot compute slope: denominator is zero".to_string());
        }
        
        let slope = numerator / denominator;
        let intercept = y_mean - slope * x_mean;
        
        self.slope = Some(slope);
        self.intercept = Some(intercept);
        self.trained = true;
        
        println!("📈 Trained Linear Regression: y = {:.4}x + {:.4}", slope, intercept);
        Ok(())
    }
    
    pub fn predict(&self, x: &[f64]) -> Result<Vec<f64>, String> {
        if !self.trained {
            return Err("Model not trained yet".to_string());
        }
        
        let slope = self.slope.unwrap();
        let intercept = self.intercept.unwrap();
        
        let predictions: Vec<f64> = x.iter()
            .map(|&x_val| slope * x_val + intercept)
            .collect();
        
        Ok(predictions)
    }
    
    pub fn score(&self, x: &[f64], y: &[f64]) -> Result<f64, String> {
        if !self.trained {
            return Err("Model not trained yet".to_string());
        }
        
        if x.len() != y.len() {
            return Err("X and y must have same length".to_string());
        }
        
        let predictions = self.predict(x)?;
        
        let y_mean: f64 = y.iter().sum::<f64>() / y.len() as f64;
        
        let mut ss_res = 0.0;
        let mut ss_tot = 0.0;
        
        for i in 0..y.len() {
            let residual = y[i] - predictions[i];
            ss_res += residual * residual;
            
            let total_diff = y[i] - y_mean;
            ss_tot += total_diff * total_diff;
        }
        
        if ss_tot.abs() < f64::EPSILON {
            return Ok(1.0);
        }
        
        let r2 = 1.0 - (ss_res / ss_tot);
        Ok(r2)
    }
    
    pub fn get_params(&self) -> Option<(f64, f64)> {
        if self.trained {
            Some((self.slope.unwrap(), self.intercept.unwrap()))
        } else {
            None
        }
    }
    
    pub fn save(&self, filename: &str) -> Result<(), String> {
        let json = serde_json::to_string_pretty(self)
            .map_err(|e| format!("Failed to serialize model: {}", e))?;
        
        std::fs::write(filename, json)
            .map_err(|e| format!("Failed to write file {}: {}", filename, e))?;
        
        println!("💾 Saved model to {}", filename);
        Ok(())
    }
    
    pub fn load(filename: &str) -> Result<Self, String> {
        let content = std::fs::read_to_string(filename)
            .map_err(|e| format!("Failed to read file {}: {}", filename, e))?;
        
        let model: LinearRegression = serde_json::from_str(&content)
            .map_err(|e| format!("Failed to parse model JSON: {}", e))?;
        
        println!("📂 Loaded model from {}", filename);
        if model.trained {
            println!("   Parameters: y = {:.4}x + {:.4}", 
                model.slope.unwrap(), model.intercept.unwrap());
        }
        Ok(model)
    }
}

/// Trainable Linear Regression for pipelines
#[derive(Debug)]
pub struct TrainableLinearRegression {
    model: LinearRegression,
    x_train: Vec<f64>,
    y_train: Vec<f64>,
}

impl TrainableLinearRegression {
    pub fn new(x_train: Vec<f64>, y_train: Vec<f64>) -> Self {
        TrainableLinearRegression {
            model: LinearRegression::new(),
            x_train,
            y_train,
        }
    }
}

impl PipelineStep for TrainableLinearRegression {
    type Input = Vec<f64>;
    type Output = Vec<f64>;
    type Error = String;

    fn process(&mut self, input: Self::Input) -> Result<Self::Output, Self::Error> {
        if !self.model.trained {
            self.model.train(&self.x_train, &self.y_train)?;
        }
        
        self.model.predict(&input)
    }
}

// ===== K-MEANS CLUSTERING =====
#[derive(Debug, serde::Serialize, serde::Deserialize)]
pub struct KMeans {
    k: usize,
    centroids: Option<Vec<f64>>,
    trained: bool,
}

impl KMeans {
    pub fn new(k: usize) -> Self {
        KMeans {
            k,
            centroids: None,
            trained: false,
        }
    }
    
    pub fn train(&mut self, data: &[f64]) -> Result<(), String> {
        if data.is_empty() {
            return Err("Data is empty".to_string());
        }
        
        let mut rng = rand::thread_rng();
        let mut centroids: Vec<f64> = (0..self.k)
            .map(|_| {
                let idx = rng.gen_range(0..data.len());
                data[idx]
            })
            .collect();
        
        // Run K-Means iterations
        for _ in 0..100 {
            let mut clusters: Vec<Vec<f64>> = vec![Vec::new(); self.k];
            
            for &point in data {
                let mut best_distance = f64::INFINITY;
                let mut best_cluster = 0;
                
                for (i, &centroid) in centroids.iter().enumerate() {
                    let distance = (point - centroid).abs();
                    if distance < best_distance {
                        best_distance = distance;
                        best_cluster = i;
                    }
                }
                
                clusters[best_cluster].push(point);
            }
            
            // Update centroids
            let new_centroids: Vec<f64> = clusters.iter()
                .map(|cluster| {
                    if cluster.is_empty() {
                        0.0
                    } else {
                        cluster.iter().sum::<f64>() / cluster.len() as f64
                    }
                })
                .collect();
            
            centroids = new_centroids;
        }
        
        self.centroids = Some(centroids);
        self.trained = true;
        
        println!("🎯 Trained K-Means with k={}", self.k);
        Ok(())
    }
    
    pub fn predict(&self, data: &[f64]) -> Result<Vec<usize>, String> {
        if !self.trained {
            return Err("Model not trained yet".to_string());
        }
        
        let centroids = self.centroids.as_ref().unwrap();
        let mut labels = Vec::with_capacity(data.len());
        
        for &point in data {
            let mut best_distance = f64::INFINITY;
            let mut best_cluster = 0;
            
            for (i, &centroid) in centroids.iter().enumerate() {
                let distance = (point - centroid).abs();
                if distance < best_distance {
                    best_distance = distance;
                    best_cluster = i;
                }
            }
            
            labels.push(best_cluster);
        }
        
        Ok(labels)
    }
    
    pub fn save(&self, filename: &str) -> Result<(), String> {
        let json = serde_json::to_string_pretty(self)
            .map_err(|e| format!("Failed to serialize model: {}", e))?;
        
        std::fs::write(filename, json)
            .map_err(|e| format!("Failed to write file {}: {}", filename, e))?;
        
        println!("💾 Saved K-Means model to {}", filename);
        Ok(())
    }
    
    pub fn load(filename: &str) -> Result<Self, String> {
        let content = std::fs::read_to_string(filename)
            .map_err(|e| format!("Failed to read file {}: {}", filename, e))?;
        
        let model: KMeans = serde_json::from_str(&content)
            .map_err(|e| format!("Failed to parse model JSON: {}", e))?;
        
        println!("📂 Loaded K-Means model from {}", filename);
        Ok(model)
    }
}

// ===== DECISION TREE =====
#[derive(Debug, serde::Serialize, serde::Deserialize)]
pub enum TreeNode {
    Leaf {
        value: f64,
        samples: usize,
    },
    Split {
        feature_index: usize,
        threshold: f64,
        left: Box<TreeNode>,
        right: Box<TreeNode>,
    },
}

#[derive(Debug, serde::Serialize, serde::Deserialize, Default)]
pub struct DecisionTree {
    max_depth: usize,
    min_samples_split: usize,
    tree: Option<TreeNode>,
    trained: bool,
}

impl DecisionTree {
    pub fn new() -> Self {
        DecisionTree {
            max_depth: 5,
            min_samples_split: 2,
            tree: None,
            trained: false,
        }
    }
    
    pub fn with_params(max_depth: usize, min_samples_split: usize) -> Self {
        DecisionTree {
            max_depth,
            min_samples_split,
            tree: None,
            trained: false,
        }
    }
    
    pub fn train(&mut self, x: &[Vec<f64>], y: &[f64]) -> Result<(), String> {
        if x.is_empty() || x.len() != y.len() {
            return Err("X and y must have same non-zero length".to_string());
        }
        
        let dataset: Vec<(Vec<f64>, f64)> = x.iter()
            .zip(y.iter())
            .map(|(features, &target)| (features.clone(), target))
            .collect();
        
        self.tree = Some(self.build_tree(&dataset, 0));
        self.trained = true;
        
        println!("🌳 Trained Decision Tree (max_depth={})", self.max_depth);
        Ok(())
    }
    
    fn build_tree(&self, dataset: &[(Vec<f64>, f64)], depth: usize) -> TreeNode {
        if dataset.len() < self.min_samples_split || depth >= self.max_depth {
            return self.create_leaf(dataset);
        }
        
        if let Some((best_feature, best_threshold)) = self.find_best_split(dataset) {
            let (left_data, right_data) = self.split_dataset(dataset, best_feature, best_threshold);
            
            if left_data.is_empty() || right_data.is_empty() {
                return self.create_leaf(dataset);
            }
            
            TreeNode::Split {
                feature_index: best_feature,
                threshold: best_threshold,
                left: Box::new(self.build_tree(&left_data, depth + 1)),
                right: Box::new(self.build_tree(&right_data, depth + 1)),
            }
        } else {
            self.create_leaf(dataset)
        }
    }
    
    fn find_best_split(&self, dataset: &[(Vec<f64>, f64)]) -> Option<(usize, f64)> {
        let n_features = dataset[0].0.len();
        let mut best_gain = -1.0;
        let mut best_split = None;
        
        for feature_idx in 0..n_features {
            let mut values: Vec<f64> = dataset.iter()
                .map(|(features, _)| features[feature_idx])
                .collect();
            
            values.sort_by_key(|&v| OrderedFloat(v));
            
            for i in 1..values.len() {
                if (values[i] - values[i-1]).abs() > f64::EPSILON {
                    let threshold = (values[i-1] + values[i]) / 2.0;
                    let (left, right) = self.split_dataset(dataset, feature_idx, threshold);
                    
                    if left.len() >= 2 && right.len() >= 2 {
                        let gain = 1.0; // Simplified gain for demo
                        if gain > best_gain {
                            best_gain = gain;
                            best_split = Some((feature_idx, threshold));
                        }
                    }
                }
            }
        }
        
        best_split
    }
    
    fn split_dataset(&self, dataset: &[(Vec<f64>, f64)], feature_idx: usize, threshold: f64) -> SplitResult {
        let mut left = Vec::new();
        let mut right = Vec::new();
        
        for (features, target) in dataset {
            if features[feature_idx] <= threshold {
                left.push((features.clone(), *target));
            } else {
                right.push((features.clone(), *target));
            }
        }
        
        (left, right)
    }
    
    fn create_leaf(&self, dataset: &[(Vec<f64>, f64)]) -> TreeNode {
        if dataset.is_empty() {
            return TreeNode::Leaf { value: 0.0, samples: 0 };
        }
        
        let value: f64 = dataset.iter()
            .map(|(_, target)| target)
            .sum::<f64>() / dataset.len() as f64;
        
        TreeNode::Leaf {
            value,
            samples: dataset.len(),
        }
    }
    
    pub fn predict(&self, x: &[Vec<f64>]) -> Result<Vec<f64>, String> {
        if !self.trained {
            return Err("Model not trained yet".to_string());
        }
        
        let tree = self.tree.as_ref().unwrap();
        let predictions: Vec<f64> = x.iter()
            .map(|features| self.predict_single(tree, features))
            .collect();
        
        Ok(predictions)
    }
    
    fn predict_single(&self, node: &TreeNode, features: &[f64]) -> f64 {
        match node {
            TreeNode::Leaf { value, .. } => *value,
            TreeNode::Split { feature_index, threshold, left, right } => {
                if features[*feature_index] <= *threshold {
                    self.predict_single(left, features)
                } else {
                    self.predict_single(right, features)
                }
            }
        }
    }
    
    pub fn save(&self, filename: &str) -> Result<(), String> {
        let json = serde_json::to_string_pretty(self)
            .map_err(|e| format!("Failed to serialize model: {}", e))?;
        
        std::fs::write(filename, json)
            .map_err(|e| format!("Failed to write file {}: {}", filename, e))?;
        
        println!("💾 Saved Decision Tree to {}", filename);
        Ok(())
    }
    
    pub fn load(filename: &str) -> Result<Self, String> {
        let content = std::fs::read_to_string(filename)
            .map_err(|e| format!("Failed to read file {}: {}", filename, e))?;
        
        let model: DecisionTree = serde_json::from_str(&content)
            .map_err(|e| format!("Failed to parse model JSON: {}", e))?;
        
        println!("📂 Loaded Decision Tree from {}", filename);
        Ok(model)
    }
}

// ===== PERCEPTRON =====
#[derive(Debug, serde::Serialize, serde::Deserialize)]
pub struct Perceptron {
    weights: Vec<f64>,
    bias: f64,
    learning_rate: f64,
    trained: bool,
}

impl Perceptron {
    pub fn new(input_size: usize) -> Self {
        let mut rng = rand::thread_rng();
        let weights: Vec<f64> = (0..input_size)
            .map(|_| rng.gen_range(-1.0..1.0))
            .collect();
        
        Perceptron {
            weights,
            bias: rng.gen_range(-1.0..1.0),
            learning_rate: 0.1,
            trained: false,
        }
    }
    
    pub fn train(&mut self, x: &[Vec<f64>], y: &[f64], epochs: usize) -> Result<(), String> {
        if x.is_empty() || x.len() != y.len() {
            return Err("X and y must have same non-zero length".to_string());
        }
        
        for epoch in 0..epochs {
            let mut total_error = 0.0;
            
            for (features, &target) in x.iter().zip(y.iter()) {
                let prediction = self.predict_single(features);
                let error = target - prediction;
                total_error += error.abs();
                
                // Update weights using iterator
                for (i, weight) in self.weights.iter_mut().enumerate() {
                    if i < features.len() {
                        *weight += self.learning_rate * error * features[i];
                    }
                }
                self.bias += self.learning_rate * error;
            }
            
            if epoch % 100 == 0 {
                println!("🧠 Perceptron epoch {}/{} - Avg error: {:.4}", 
                    epoch, epochs, total_error / x.len() as f64);
            }
        }
        
        self.trained = true;
        println!("✅ Trained Perceptron with {} weights", self.weights.len());
        Ok(())
    }
    
    pub fn predict(&self, x: &[Vec<f64>]) -> Result<Vec<f64>, String> {
        if !self.trained {
            return Err("Model not trained yet".to_string());
        }
        
        let predictions: Vec<f64> = x.iter()
            .map(|features| self.predict_single(features))
            .collect();
        
        Ok(predictions)
    }
    
    fn predict_single(&self, features: &[f64]) -> f64 {
        let mut sum = self.bias;
        for (i, &weight) in self.weights.iter().enumerate() {
            if i < features.len() {
                sum += weight * features[i];
            }
        }
        // Sigmoid activation
        1.0 / (1.0 + (-sum).exp())
    }
    
    pub fn save(&self, filename: &str) -> Result<(), String> {
        let json = serde_json::to_string_pretty(self)
            .map_err(|e| format!("Failed to serialize model: {}", e))?;
        
        std::fs::write(filename, json)
            .map_err(|e| format!("Failed to write file {}: {}", filename, e))?;
        
        println!("💾 Saved Perceptron to {}", filename);
        Ok(())
    }
    
    pub fn load(filename: &str) -> Result<Self, String> {
        let content = std::fs::read_to_string(filename)
            .map_err(|e| format!("Failed to read file {}: {}", filename, e))?;
        
        let model: Perceptron = serde_json::from_str(&content)
            .map_err(|e| format!("Failed to parse model JSON: {}", e))?;
        
        println!("📂 Loaded Perceptron from {}", filename);
        Ok(model)
    }
}

// ===== HELPER FUNCTIONS =====
pub fn parse_comma_separated(input: &str) -> Result<Vec<f64>, String> {
    let mut values = Vec::new();
    
    for part in input.split(',') {
        let trimmed = part.trim();
        if !trimmed.is_empty() {
            match trimmed.parse::<f64>() {
                Ok(value) => values.push(value),
                Err(_) => return Err(format!("Invalid number: '{}'", trimmed)),
            }
        }
    }
    
    if values.is_empty() {
        Err("No valid numbers found".to_string())
    } else {
        Ok(values)
    }
}

pub fn create_demo_pipeline() -> Pipeline {
    let mut pipeline = Pipeline::new();
    
    let x_train = vec![1.0, 2.0, 3.0, 4.0];
    let y_train = vec![3.0, 5.0, 7.0, 9.0];
    
    pipeline.add_step(Box::new(TrainableLinearRegression::new(x_train, y_train)));
    pipeline
}