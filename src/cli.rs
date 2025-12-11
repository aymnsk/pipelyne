// src/cli.rs - FINAL POLISHED VERSION
use clap::{Parser, Subcommand};

#[derive(Parser)]
#[command(name = "pipelyne")]
#[command(about = "Machine Learning Pipeline Framework", long_about = None)]
#[command(version = "1.0")]
#[command(author = "ML Pipeline Team")]
pub struct Cli {
    #[command(subcommand)]
    pub command: Commands,
}

#[derive(Subcommand)]
pub enum Commands {
    /// Train a new machine learning model
    Train {
        /// Input CSV file with training data
        #[arg(short = 'd', long = "data")]
        data: String,
        
        /// Model type to train
        #[arg(short = 'm', long = "model", 
               value_parser = ["linear_regression", "kmeans", "decision_tree", "perceptron"],
               default_value = "linear_regression")]
        model: String,
        
        /// Output model file (JSON format)
        #[arg(short = 'o', long = "output", default_value = "model.json")]
        output: String,
        
        /// For K-Means: number of clusters
        #[arg(short = 'k', long = "clusters")]
        k: Option<usize>,
        
        /// For Decision Tree: maximum depth
        #[arg(long = "max-depth")]
        max_depth: Option<usize>,
        
        /// For Decision Tree: minimum samples to split
        #[arg(long = "min-samples")]
        min_samples: Option<usize>,
        
        /// For Perceptron: training epochs
        #[arg(long = "epochs")]
        epochs: Option<usize>,
    },
    
    /// Make predictions with a trained model
    Predict {
        /// Trained model file (JSON format)
        #[arg(short = 'm', long = "model")]
        model: String,
        
        /// Input values (comma-separated numbers)
        #[arg(short = 'i', long = "input")]
        input: String,
        
        /// Output file for predictions (optional)
        #[arg(short = 'o', long = "output")]
        output: Option<String>,
    },
    
    /// Show pipeline information
    Info {
        /// Pipeline info file
        #[arg(short = 'f', long = "file")]
        file: String,
    },
    
    /// Run a test example
    Demo {
        /// Example to run
        #[arg(value_parser = ["regression", "kmeans", "decision_tree", "perceptron", 
                               "pipeline", "xy_data", "save_load", "all_models"],
               default_value = "regression")]
        example: String,
    },
    
    /// Save a pre-trained model to file
    Save {
        /// Model type to save
        #[arg(value_parser = ["demo", "kmeans", "decision_tree", "perceptron"],
               default_value = "demo")]
        model_type: String,
        
        /// Output filename
        #[arg(short = 'o', long = "output", default_value = "trained_model.json")]
        output: String,
        
        /// For K-Means: number of clusters
        #[arg(short = 'k', long = "clusters")]
        k: Option<usize>,
        
        /// For Decision Tree: maximum depth
        #[arg(long = "max-depth")]
        max_depth: Option<usize>,
        
        /// For Decision Tree: minimum samples to split
        #[arg(long = "min-samples")]
        min_samples: Option<usize>,
        
        /// For Perceptron: training epochs
        #[arg(long = "epochs")]
        epochs: Option<usize>,
    },
    
    /// Load and test a saved model
    Load {
        /// Model file to load
        #[arg(short = 'f', long = "file")]
        file: String,
        
        /// Input values to test (comma-separated)
        #[arg(short = 'i', long = "input")]
        input: String,
    },
}