// src/main.rs - FINAL POLISHED VERSION
mod cli;
use clap::Parser;

use pipelyne::{Pipeline, CsvLoader, StandardScaler, LinearRegression, KMeans, DecisionTree, Perceptron, XyDataLoader, TrainTestSplit, PipelineStep, parse_comma_separated, create_demo_pipeline};
use std::fs;

fn main() {
    let cli = cli::Cli::parse();
    
    match cli.command {
        cli::Commands::Train { data, model, output, k, max_depth, min_samples, epochs } => {
            println!("🚀 Training {} model with data from {}", model, data);
            
            match model.as_str() {
                "linear_regression" => {
                    println!("📊 Attempting to load X,y data from CSV...");
                    
                    let mut loader = XyDataLoader::new(&data, 0, 1);
                    match loader.process(()) {
                        Ok((x_data, y_data)) => {
                            println!("✅ Loaded {} X,y pairs", x_data.len());
                            
                            let mut lr_model = LinearRegression::new();
                            if lr_model.train(&x_data, &y_data).is_ok() {
                                if let Ok(r2) = lr_model.score(&x_data, &y_data) {
                                    println!("📊 R² score: {:.4}", r2);
                                }
                                
                                if lr_model.save(&output).is_ok() {
                                    println!("💾 Model saved to {}", output);
                                }
                            }
                        }
                        Err(e) => {
                            println!("⚠️  Could not load as X,y data: {}", e);
                            println!("📊 Falling back to demo data...");
                            
                            let mut pipeline = create_demo_pipeline();
                            if pipeline.run(vec![1.0]).is_ok() {
                                println!("💡 Try: pipelyne save --output {} to save the trained model", output);
                                if pipeline.save_info(&output).is_ok() {
                                    println!("✅ Pipeline info saved to {}", output);
                                }
                            }
                        }
                    }
                }
                
                "kmeans" => {
                    println!("🎯 Training K-Means clustering model");
                    
                    let mut loader = CsvLoader::new(&data);
                    match loader.process(vec![]) {
                        Ok(data_points) => {
                            println!("📊 Loaded {} data points", data_points.len());
                            
                            let k_value = k.unwrap_or(3);
                            let mut kmeans = KMeans::new(k_value);
                            if kmeans.train(&data_points).is_ok() {
                                if let Ok(labels) = kmeans.predict(&data_points[0..5.min(data_points.len())]) {
                                    println!("🔍 First 5 predictions: {:?}", labels);
                                    println!("   Data: {:?}", &data_points[0..5.min(data_points.len())]);
                                }
                                
                                if kmeans.save(&output).is_ok() {
                                    println!("💾 K-Means model saved to {}", output);
                                }
                            }
                        }
                        Err(e) => println!("❌ Data loading error: {}", e),
                    }
                }
                
                "decision_tree" => {
                    println!("🌳 Training Decision Tree model");
                    
                    // For decision tree, we need 2D features
                    // Let's create synthetic data or load multi-column CSV
                    println!("📊 Creating synthetic data for Decision Tree demo...");
                    
                    // Create simple dataset: if x1 > 5 AND x2 > 3 then y=1 else y=0
                    let x_train = vec![
                        vec![2.0, 1.0],
                        vec![3.0, 2.0],
                        vec![6.0, 4.0],
                        vec![7.0, 5.0],
                        vec![8.0, 2.0],
                        vec![9.0, 6.0],
                    ];
                    
                    let y_train = vec![0.0, 0.0, 1.0, 1.0, 1.0, 1.0];
                    
                    let max_depth_val = max_depth.unwrap_or(3);
                    let min_samples_val = min_samples.unwrap_or(2);
                    
                    let mut tree = DecisionTree::with_params(max_depth_val, min_samples_val);
                    if tree.train(&x_train, &y_train).is_ok() {
                        // Test predictions
                        let test_x = vec![
                            vec![1.0, 1.0],  // Should be 0
                            vec![6.0, 5.0],  // Should be 1
                            vec![4.0, 3.0],  // Should be 0
                        ];
                        
                        if let Ok(predictions) = tree.predict(&test_x) {
                            println!("🌲 Decision Tree predictions:");
                            for (features, pred) in test_x.iter().zip(predictions.iter()) {
                                println!("  Features {:?} → {:.2}", features, pred);
                            }
                        }
                        
                        if tree.save(&output).is_ok() {
                            println!("💾 Decision Tree saved to {}", output);
                        }
                    }
                }
                
                "perceptron" => {
                    println!("🧠 Training Perceptron (neural network)");
                    
                    // Create XOR dataset (non-linear problem)
                    let x_train = vec![
                        vec![0.0, 0.0],
                        vec![0.0, 1.0],
                        vec![1.0, 0.0],
                        vec![1.0, 1.0],
                    ];
                    
                    let y_train = vec![0.0, 1.0, 1.0, 0.0];  // XOR
                    
                    let epochs_val = epochs.unwrap_or(1000);
                    let mut perceptron = Perceptron::new(2);
                    
                    if perceptron.train(&x_train, &y_train, epochs_val).is_ok() {
                        // Test predictions
                        if let Ok(predictions) = perceptron.predict(&x_train) {
                            println!("🧠 Perceptron predictions (XOR problem):");
                            for (i, (features, &pred)) in x_train.iter().zip(predictions.iter()).enumerate() {
                                let expected = y_train[i];
                                println!("  {:?} → {:.3} (expected: {})", features, pred, expected);
                            }
                        }
                        
                        if perceptron.save(&output).is_ok() {
                            println!("💾 Perceptron saved to {}", output);
                        }
                    }
                }
                
                _ => println!("❌ Unknown model type: {}", model),
            }
        }
        
        cli::Commands::Predict { model, input, output } => {
            println!("🔮 Making predictions with model: {}", model);
            
            match parse_comma_separated(&input) {
                Ok(values) => {
                    println!("📊 Input values: {:?}", values);
                    
                    if model.ends_with(".json") {
                        // Try to determine model type by reading file content
                        match std::fs::read_to_string(&model) {
                            Ok(content) => {
                                // Try KMeans first (has 'k' field)
                                if let Ok(kmeans_model) = serde_json::from_str::<KMeans>(&content) {
                                    match kmeans_model.predict(&values) {
                                        Ok(labels) => {
                                            println!("🎯 K-Means cluster assignments: {:?}", labels);
                                            save_predictions_ints(&labels, output);
                                            return;
                                        }
                                        Err(_) => {}
                                    }
                                }
                                
                                // Try LinearRegression
                                if let Ok(lr_model) = serde_json::from_str::<LinearRegression>(&content) {
                                    match lr_model.predict(&values) {
                                        Ok(predictions) => {
                                            println!("📈 Linear Regression predictions: {:?}", predictions);
                                            if let Some(params) = lr_model.get_params() {
                                                println!("  Model: y = {:.4}x + {:.4}", params.0, params.1);
                                            }
                                            save_predictions(&predictions, output);
                                            return;
                                        }
                                        Err(_) => {}
                                    }
                                }
                                
                                // Try DecisionTree
                                if let Ok(tree_model) = serde_json::from_str::<DecisionTree>(&content) {
                                    let features = vec![values.clone()];
                                    match tree_model.predict(&features) {
                                        Ok(predictions) => {
                                            println!("🌳 Decision Tree predictions: {:?}", predictions);
                                            save_predictions(&predictions, output);
                                            return;
                                        }
                                        Err(_) => {}
                                    }
                                }
                                
                                // Try Perceptron
                                if let Ok(perceptron_model) = serde_json::from_str::<Perceptron>(&content) {
                                    let features = vec![values.clone()];
                                    match perceptron_model.predict(&features) {
                                        Ok(predictions) => {
                                            println!("🧠 Perceptron predictions: {:?}", predictions);
                                            save_predictions(&predictions, output);
                                            return;
                                        }
                                        Err(_) => {}
                                    }
                                }
                                
                                println!("❌ Could not determine model type in {}", model);
                            }
                            Err(e) => println!("❌ Failed to read model file: {}", e),
                        }
                    } else {
                        let mut pipeline = create_demo_pipeline();
                        match pipeline.run(values) {
                            Ok(predictions) => {
                                println!("✅ Predictions: {:?}", predictions);
                                save_predictions(&predictions, output);
                            }
                            Err(e) => println!("❌ Prediction error: {}", e),
                        }
                    }
                }
                Err(e) => println!("❌ Invalid input format: {}", e),
            }
        }
        
        cli::Commands::Info { file } => {
            println!("📄 Pipeline information: {}", file);
            
            match Pipeline::load_info(&file) {
                Ok(info) => {
                    println!("📊 Pipeline Summary:");
                    println!("  Steps: {}", info.num_steps);
                    for (i, step_type) in info.step_types.iter().enumerate() {
                        println!("  Step {}: {}", i + 1, step_type);
                    }
                }
                Err(e) => println!("❌ Error: {}", e),
            }
        }
        
        cli::Commands::Demo { example } => {
            println!("🎮 Running demo: {}", example);
            
            match example.as_str() {
                "regression" => {
                    println!("\n=== Linear Regression Demo ===");
                    
                    let mut pipeline = create_demo_pipeline();
                    
                    println!("Training on: y = 2x + 1");
                    println!("  X: [1, 2, 3, 4]");
                    println!("  y: [3, 5, 7, 9]");
                    
                    match pipeline.run(vec![5.0, 6.0, 7.0]) {
                        Ok(predictions) => {
                            println!("\nPredicting for [5, 6, 7]:");
                            println!("✅ Predictions: {:?}", predictions);
                            println!("  Expected: [11, 13, 15]");
                        }
                        Err(e) => println!("❌ Error: {}", e),
                    }
                }
                
                "save_load" => {
                    println!("\n=== Save/Load Model Demo ===");
                    
                    let mut model = LinearRegression::new();
                    let x = vec![1.0, 2.0, 3.0, 4.0];
                    let y = vec![3.0, 5.0, 7.0, 9.0];
                    
                    match model.train(&x, &y) {
                        Ok(_) => {
                            match model.save("demo_lr_model.json") {
                                Ok(_) => {
                                    println!("💾 Model saved to demo_lr_model.json");
                                    
                                    match LinearRegression::load("demo_lr_model.json") {
                                        Ok(loaded_model) => {
                                            match loaded_model.predict(&[8.0, 9.0]) {
                                                Ok(predictions) => {
                                                    println!("📊 Loaded model predictions for [8, 9]:");
                                                    println!("✅ Predictions: {:?}", predictions);
                                                    println!("  Expected: [17, 19]");
                                                }
                                                Err(e) => println!("❌ Prediction error: {}", e),
                                            }
                                        }
                                        Err(e) => println!("❌ Load error: {}", e),
                                    }
                                }
                                Err(e) => println!("❌ Save error: {}", e),
                            }
                        }
                        Err(e) => println!("❌ Training error: {}", e),
                    }
                }
                
                "kmeans" => {
                    println!("\n=== K-Means Clustering Demo ===");
                    
                    let data = vec![
                        1.0, 1.2, 1.3, 1.1,
                        5.0, 5.5, 4.8, 5.2,
                        9.0, 9.1, 8.9, 9.3,
                    ];
                    
                    println!("Data: {:?}", data);
                    println!("Expected: 3 clusters around 1.0, 5.0, 9.0");
                    
                    let mut kmeans = KMeans::new(3);
                    match kmeans.train(&data) {
                        Ok(_) => {
                            match kmeans.predict(&data) {
                                Ok(labels) => {
                                    println!("🎯 Cluster assignments:");
                                    for (i, (&point, &label)) in data.iter().zip(labels.iter()).enumerate() {
                                        println!("  Point {}: {:.1} → Cluster {}", i, point, label);
                                    }
                                    
                                    match kmeans.save("demo_kmeans_model.json") {
                                        Ok(_) => println!("💾 K-Means model saved to demo_kmeans_model.json"),
                                        Err(e) => println!("❌ Save error: {}", e),
                                    }
                                }
                                Err(e) => println!("❌ Prediction error: {}", e),
                            }
                        }
                        Err(e) => println!("❌ Training error: {}", e),
                    }
                }
                
                "decision_tree" => {
                    println!("\n=== Decision Tree Demo ===");
                    
                    // Simple dataset: if temperature > 25 and humidity > 60, rain = 1
                    let x = vec![
                        vec![20.0, 50.0],  // Cool, dry
                        vec![30.0, 70.0],  // Hot, humid
                        vec![15.0, 80.0],  // Cool, humid
                        vec![28.0, 65.0],  // Warm, humid
                    ];
                    
                    let y = vec![0.0, 1.0, 0.0, 1.0];  // Rain or not
                    
                    println!("🌡️  Weather prediction dataset:");
                    for (i, features) in x.iter().enumerate() {
                        println!("  Day {}: Temp={}, Humidity={} → Rain={}", 
                            i+1, features[0], features[1], y[i]);
                    }
                    
                    let mut tree = DecisionTree::with_params(3, 2);
                    match tree.train(&x, &y) {
                        Ok(_) => {
                            let test_cases = vec![
                                vec![22.0, 55.0],  // Should be 0
                                vec![32.0, 75.0],  // Should be 1
                                vec![18.0, 40.0],  // Should be 0
                            ];
                            
                            match tree.predict(&test_cases) {
                                Ok(predictions) => {
                                    println!("🌳 Decision Tree predictions:");
                                    for (i, (features, &pred)) in test_cases.iter().zip(predictions.iter()).enumerate() {
                                        println!("  Case {}: {:?} → Rain probability: {:.2}", 
                                            i+1, features, pred);
                                    }
                                }
                                Err(e) => println!("❌ Prediction error: {}", e),
                            }
                            
                            match tree.save("demo_tree_model.json") {
                                Ok(_) => println!("💾 Decision Tree saved to demo_tree_model.json"),
                                Err(e) => println!("❌ Save error: {}", e),
                            }
                        }
                        Err(e) => println!("❌ Training error: {}", e),
                    }
                }
                
                "perceptron" => {
                    println!("\n=== Perceptron (Neural Network) Demo ===");
                    
                    // XOR problem (non-linear)
                    let x = vec![
                        vec![0.0, 0.0],
                        vec![0.0, 1.0],
                        vec![1.0, 0.0],
                        vec![1.0, 1.0],
                    ];
                    
                    let y = vec![0.0, 1.0, 1.0, 0.0];  // XOR
                    
                    println!("🧠 XOR Problem (non-linear):");
                    println!("  0 XOR 0 = 0");
                    println!("  0 XOR 1 = 1");
                    println!("  1 XOR 0 = 1");
                    println!("  1 XOR 1 = 0");
                    
                    let mut perceptron = Perceptron::new(2);
                    match perceptron.train(&x, &y, 2000) {
                        Ok(_) => {
                            match perceptron.predict(&x) {
                                Ok(predictions) => {
                                    println!("🧠 Perceptron predictions:");
                                    for (i, (features, &pred)) in x.iter().zip(predictions.iter()).enumerate() {
                                        let expected = y[i];
                                        let rounded = if pred > 0.5 { 1.0 } else { 0.0 };
                                        println!("  {:?} → {:.3} (rounded: {:.0}, expected: {:.0})", 
                                            features, pred, rounded, expected);
                                    }
                                }
                                Err(e) => println!("❌ Prediction error: {}", e),
                            }
                            
                            match perceptron.save("demo_perceptron_model.json") {
                                Ok(_) => println!("💾 Perceptron saved to demo_perceptron_model.json"),
                                Err(e) => println!("❌ Save error: {}", e),
                            }
                        }
                        Err(e) => println!("❌ Training error: {}", e),
                    }
                }
                
                "pipeline" => {
                    println!("\n=== Multi-step Pipeline Demo ===");
                    
                    let mut pipeline = Pipeline::new();
                    pipeline.add_step(Box::new(CsvLoader::new("data.csv")));
                    pipeline.add_step(Box::new(StandardScaler::new()));
                    
                    println!("Running: Load CSV → Standardize");
                    
                    match pipeline.run(vec![]) {
                        Ok(result) => {
                            println!("✅ Result: {:?}", result);
                            println!("  (Data normalized to mean=0, std=1)");
                        }
                        Err(e) => println!("❌ Error: {}", e),
                    }
                }
                
                "xy_data" => {
                    println!("\n=== X,y Data Processing Demo ===");
                    
                    let csv_content = "x,y\n1,3\n2,5\n3,7\n4,9\n5,11\n";
                    fs::write("xy_data.csv", csv_content).unwrap();
                    println!("📝 Created xy_data.csv with 5 X,y pairs");
                    
                    let mut loader = XyDataLoader::new("xy_data.csv", 0, 1);
                    match loader.process(()) {
                        Ok((x_data, y_data)) => {
                            println!("✅ Loaded {} X,y pairs", x_data.len());
                            println!("  X: {:?}", x_data);
                            println!("  y: {:?}", y_data);
                            
                            let mut splitter = TrainTestSplit::new(0.3);
                            match splitter.process((x_data, y_data)) {
                                Ok((x_train, x_test, y_train, y_test)) => {
                                    println!("📊 Train/Test Split:");
                                    println!("  X_train ({}): {:?}", x_train.len(), x_train);
                                    println!("  X_test ({}): {:?}", x_test.len(), x_test);
                                    
                                    let mut model = LinearRegression::new();
                                    match model.train(&x_train, &y_train) {
                                        Ok(_) => {
                                            println!("📈 Trained on {} samples", x_train.len());
                                            
                                            match model.predict(&x_test) {
                                                Ok(predictions) => {
                                                    println!("🔍 Test predictions: {:?}", predictions);
                                                    println!("  Actual y_test: {:?}", y_test);
                                                    
                                                    match model.score(&x_test, &y_test) {
                                                        Ok(r2) => println!("📊 Test R² score: {:.4}", r2),
                                                        Err(e) => println!("⚠️  Could not calculate score: {}", e),
                                                    }
                                                }
                                                Err(e) => println!("❌ Prediction error: {}", e),
                                            }
                                        }
                                        Err(e) => println!("❌ Training error: {}", e),
                                    }
                                }
                                Err(e) => println!("❌ Split error: {}", e),
                            }
                        }
                        Err(e) => println!("❌ Load error: {}", e),
                    }
                }
                
                "all_models" => {
                    println!("\n=== All Models Comparison Demo ===");
                    
                    // Simple dataset
                    let x = vec![1.0, 2.0, 3.0, 4.0, 5.0];
                    let y = vec![2.0, 4.0, 5.0, 4.0, 5.0];  // Not perfectly linear
                    
                    println!("📊 Dataset: {:?}", x.iter().zip(y.iter())
                        .map(|(&xi, &yi)| format!("({},{})", xi, yi))
                        .collect::<Vec<_>>()
                        .join(" "));
                    
                    // 1. Linear Regression
                    println!("\n1. 📈 Linear Regression:");
                    let mut lr = LinearRegression::new();
                    if lr.train(&x, &y).is_ok() {
                        if let Ok(pred) = lr.predict(&[6.0]) {
                            println!("   Prediction for x=6: {:.2}", pred[0]);
                        }
                        if let Ok(r2) = lr.score(&x, &y) {
                            println!("   R² score: {:.4}", r2);
                        }
                    }
                    
                    // 2. Decision Tree (convert to 2D)
                    println!("\n2. 🌳 Decision Tree:");
                    let x_2d: Vec<Vec<f64>> = x.iter().map(|&v| vec![v]).collect();
                    let mut tree = DecisionTree::with_params(3, 2);
                    if tree.train(&x_2d, &y).is_ok() {
                        if let Ok(pred) = tree.predict(&[vec![6.0]]) {
                            println!("   Prediction for x=6: {:.2}", pred[0]);
                        }
                    }
                    
                    // 3. K-Means Clustering
                    println!("\n3. 🎯 K-Means Clustering:");
                    let mut kmeans = KMeans::new(3);
                    if kmeans.train(&x).is_ok() {
                        if let Ok(labels) = kmeans.predict(&x) {
                            println!("   Cluster assignments: {:?}", labels);
                        }
                    }
                    
                    // 4. Perceptron
                    println!("\n4. 🧠 Perceptron:");
                    let mut perceptron = Perceptron::new(1);
                    if perceptron.train(&x_2d, &y, 500).is_ok() {
                        if let Ok(pred) = perceptron.predict(&[vec![6.0]]) {
                            println!("   Prediction for x=6: {:.3}", pred[0]);
                        }
                    }
                    
                    println!("\n✅ Different models for different problems!");
                    println!("   • Linear: Best for linear relationships");
                    println!("   • Decision Tree: Handles non-linear, categorical");
                    println!("   • K-Means: Unsupervised clustering");
                    println!("   • Perceptron: Neural network basics");
                }
                
                _ => println!("❌ Unknown demo: {}. Try 'regression', 'kmeans', 'decision_tree', 'perceptron', 'pipeline', 'xy_data', or 'all_models'", example),
            }
        }
        
        cli::Commands::Save { model_type, output, k, max_depth, min_samples, epochs } => {
            println!("💾 Saving {} model to {}", model_type, output);
            
            match model_type.as_str() {
                "demo" => {
                    let mut model = LinearRegression::new();
                    let x = vec![1.0, 2.0, 3.0, 4.0];
                    let y = vec![3.0, 5.0, 7.0, 9.0];
                    
                    match model.train(&x, &y) {
                        Ok(_) => {
                            if let Ok(r2) = model.score(&x, &y) {
                                println!("📊 R² score: {:.4}", r2);
                            }
                            
                            if model.save(&output).is_ok() {
                                println!("✅ Linear Regression model saved to {}", output);
                            }
                        }
                        Err(e) => println!("❌ Training error: {}", e),
                    }
                }
                
                "kmeans" => {
                    let data = vec![1.0, 1.1, 1.2, 5.0, 5.1, 9.0, 9.2];
                    let k_value = k.unwrap_or(3);
                    let mut kmeans = KMeans::new(k_value);
                    
                    if kmeans.train(&data).is_ok() {
                        if kmeans.save(&output).is_ok() {
                            println!("✅ K-Means model (k={}) saved to {}", k_value, output);
                        }
                    }
                }
                
                "decision_tree" => {
                    let x = vec![
                        vec![1.0, 2.0],
                        vec![2.0, 3.0],
                        vec![3.0, 4.0],
                        vec![4.0, 5.0],
                    ];
                    let y = vec![0.0, 0.0, 1.0, 1.0];
                    
                    let max_depth_val = max_depth.unwrap_or(3);
                    let min_samples_val = min_samples.unwrap_or(2);
                    
                    let mut tree = DecisionTree::with_params(max_depth_val, min_samples_val);
                    
                    if tree.train(&x, &y).is_ok() {
                        if tree.save(&output).is_ok() {
                            println!("✅ Decision Tree saved to {}", output);
                        }
                    }
                }
                
                "perceptron" => {
                    let x = vec![
                        vec![0.0, 0.0],
                        vec![0.0, 1.0],
                        vec![1.0, 0.0],
                        vec![1.0, 1.0],
                    ];
                    let y = vec![0.0, 1.0, 1.0, 0.0];
                    
                    let epochs_val = epochs.unwrap_or(1000);
                    let mut perceptron = Perceptron::new(2);
                    
                    if perceptron.train(&x, &y, epochs_val).is_ok() {
                        if perceptron.save(&output).is_ok() {
                            println!("✅ Perceptron saved to {}", output);
                        }
                    }
                }
                
                _ => println!("❌ Unknown model type: {}", model_type),
            }
        }
        
        cli::Commands::Load { file, input } => {
            println!("📂 Loading model from {}", file);
            
            match parse_comma_separated(&input) {
                Ok(values) => {
                    println!("📊 Input values: {:?}", values);
                    
                    // Try to load based on file content
                    match std::fs::read_to_string(&file) {
                        Ok(content) => {
                            // Try KMeans first
                            if let Ok(kmeans_model) = serde_json::from_str::<KMeans>(&content) {
                                match kmeans_model.predict(&values) {
                                    Ok(labels) => {
                                        println!("🎯 K-Means cluster assignments: {:?}", labels);
                                    }
                                    Err(e) => println!("❌ K-Means prediction error: {}", e),
                                }
                            } else if let Ok(lr_model) = serde_json::from_str::<LinearRegression>(&content) {
                                match lr_model.predict(&values) {
                                    Ok(predictions) => {
                                        println!("📈 Linear Regression predictions: {:?}", predictions);
                                        if let Some(params) = lr_model.get_params() {
                                            println!("  Model: y = {:.4}x + {:.4}", params.0, params.1);
                                        }
                                    }
                                    Err(e) => println!("❌ Linear Regression prediction error: {}", e),
                                }
                            } else if let Ok(tree_model) = serde_json::from_str::<DecisionTree>(&content) {
                                let features = vec![values.clone()];
                                match tree_model.predict(&features) {
                                    Ok(predictions) => {
                                        println!("🌳 Decision Tree predictions: {:?}", predictions);
                                    }
                                    Err(e) => println!("❌ Decision Tree prediction error: {}", e),
                                }
                            } else if let Ok(perceptron_model) = serde_json::from_str::<Perceptron>(&content) {
                                let features = vec![values.clone()];
                                match perceptron_model.predict(&features) {
                                    Ok(predictions) => {
                                        println!("🧠 Perceptron predictions: {:?}", predictions);
                                    }
                                    Err(e) => println!("❌ Perceptron prediction error: {}", e),
                                }
                            } else {
                                println!("❌ Could not determine model type in {}", file);
                            }
                        }
                        Err(e) => println!("❌ Load error: {}", e),
                    }
                }
                Err(e) => println!("❌ Invalid input format: {}", e),
            }
        }
    }
}

fn save_predictions(predictions: &[f64], output: Option<String>) {
    if let Some(output_file) = output {
        let output_str = predictions.iter()
            .map(|v| v.to_string())
            .collect::<Vec<_>>()
            .join("\n");
        
        match fs::write(&output_file, output_str) {
            Ok(_) => println!("💾 Predictions saved to {}", output_file),
            Err(e) => println!("❌ Save error: {}", e),
        }
    }
}

fn save_predictions_ints(predictions: &[usize], output: Option<String>) {
    if let Some(output_file) = output {
        let output_str = predictions.iter()
            .map(|v| v.to_string())
            .collect::<Vec<_>>()
            .join("\n");
        
        match fs::write(&output_file, output_str) {
            Ok(_) => println!("💾 Predictions saved to {}", output_file),
            Err(e) => println!("❌ Save error: {}", e),
        }
    }
}