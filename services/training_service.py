"""
Training Service for managing parallel execution of Sub Training Agents
"""
import asyncio
import json
from typing import List, Dict, Any, Optional
from agents import Runner
from types import SimpleNamespace

class TrainingService:
    """Service for managing the Master Training Agent and Sub Training Agents"""
    
    def __init__(self):
        self.sub_agents = []
        # Import locally to avoid circular dependency
        from services.agent_service import create_evaluator_agent
        self.evaluator_agent = create_evaluator_agent()
    
    async def analyze_dataset_characteristics(self, dataset_path: str, target_column: str, ctx: SimpleNamespace) -> Dict[str, Any]:
        """Analyze dataset to determine characteristics for model selection"""
        
        analysis_script = f"""
                            import pandas as pd
                            import numpy as np
                            from sklearn.model_selection import train_test_split
                            from sklearn.preprocessing import LabelEncoder
                            import json

                            # Load dataset
                            df = pd.read_csv('{dataset_path}')

                            # Basic characteristics
                            size = len(df)
                            feature_count = len(df.columns) - 1  # Exclude target column
                            target_column = '{target_column}'

                            # Problem type detection
                            if df[target_column].dtype == 'object' or df[target_column].dtype.name == 'category':
                                problem_type = 'classification'
                                unique_classes = df[target_column].nunique()
                                class_distribution = df[target_column].value_counts().to_dict()
                                
                                # Check if balanced
                                class_counts = list(class_distribution.values())
                                max_class = max(class_counts)
                                min_class = min(class_counts)
                                balance_ratio = min_class / max_class
                                is_balanced = balance_ratio > 0.3
                            else:
                                problem_type = 'regression'
                                unique_classes = None
                                class_distribution = None
                                is_balanced = None

                            # Feature types analysis
                            numerical_features = df.select_dtypes(include=[np.number]).columns.tolist()
                            categorical_features = df.select_dtypes(include=['object', 'category']).columns.tolist()

                            # Remove target column from feature analysis
                            if target_column in numerical_features:
                                numerical_features.remove(target_column)
                            if target_column in categorical_features:
                                categorical_features.remove(target_column)

                            feature_types = {{
                                'numerical_count': len(numerical_features),
                                'categorical_count': len(categorical_features),
                                'numerical_features': numerical_features,
                                'categorical_features': categorical_features
                            }}

                            # Missing values
                            missing_values = df.isnull().sum().to_dict()

                            # Dataset size category
                            if size < 1000:
                                size_category = 'small'
                            elif size < 10000:
                                size_category = 'medium'
                            else:
                                size_category = 'large'

                            # Feature count category
                            if feature_count < 20:
                                feature_category = 'low'
                            elif feature_count < 100:
                                feature_category = 'medium'
                            else:
                                feature_category = 'high'

                            characteristics = {{
                                'size': size,
                                'size_category': size_category,
                                'feature_count': feature_count,
                                'feature_category': feature_category,
                                'problem_type': problem_type,
                                'target_column': target_column,
                                'unique_classes': unique_classes,
                                'class_distribution': class_distribution,
                                'is_balanced': is_balanced,
                                'feature_types': feature_types,
                                'missing_values': missing_values
                            }}

                            # Save characteristics
                            with open('dataset_characteristics.json', 'w') as f:
                                json.dump(characteristics, f, indent=2)

                            print("Dataset characteristics analysis complete")
                            print(f"Size: {{size}} samples, {{feature_count}} features")
                            print(f"Problem type: {{problem_type}}")
                            print(f"Target column: {{target_column}}")
                        """
        
        # Run analysis in Daytona sandbox
        from services.agent_service import daytona_run_script
        
        result = daytona_run_script(
            ctx=ctx,
            script_name="analyze_dataset.py",
            script=analysis_script,
            requirements="pandas numpy scikit-learn",
            dataset_destination=dataset_path
        )
        
        # Parse result and load characteristics
        try:
            result_data = json.loads(result)
            if result_data.get("exit_code") == 0:
                # Load the characteristics file that was created
                load_script = """
                                import json
                                with open('dataset_characteristics.json', 'r') as f:
                                    characteristics = json.load(f)
                                print(json.dumps(characteristics))
                            """
                load_result = daytona_run_script(
                    ctx=ctx,
                    script_name="load_characteristics.py",
                    script=load_script,
                    requirements=""
                )
                load_data = json.loads(load_result)
                if load_data.get("exit_code") == 0:
                    # Extract characteristics from output
                    output_lines = load_data.get("output", "").split('\n')
                    for line in output_lines:
                        if line.strip().startswith('{'):
                            return json.loads(line.strip())
        except Exception as e:
            print(f"Error parsing dataset characteristics: {e}")
        
        # Fallback characteristics
        return {
            'size': 'Unknown',
            'size_category': 'medium',
            'feature_count': 'Unknown',
            'feature_category': 'medium',
            'problem_type': 'classification',
            'target_column': target_column,
            'unique_classes': None,
            'class_distribution': None,
            'is_balanced': True,
            'feature_types': {'numerical_count': 0, 'categorical_count': 0},
            'missing_values': {}
        }
    
    def select_top_5_models(self, characteristics: Dict[str, Any]) -> List[Dict[str, str]]:
        """Select top 5 models based on dataset characteristics"""
        
        problem_type = characteristics.get('problem_type', 'classification')
        size_category = characteristics.get('size_category', 'medium')
        feature_category = characteristics.get('feature_category', 'medium')
        is_balanced = characteristics.get('is_balanced', True)
        
        # Model selection logic based on characteristics
        if problem_type == 'classification':
            if size_category == 'small':
                # Small datasets: simpler models work better
                models = [
                    {'name': 'LogisticRegression', 'category': 'Linear models'},
                    {'name': 'SVC', 'category': 'SVM and kernel methods'},
                    {'name': 'RandomForest', 'category': 'Tree-based models'},
                    {'name': 'MLPClassifier', 'category': 'Neural networks'},
                    {'name': 'AdaBoost', 'category': 'Ensemble methods'}
                ]
            elif size_category == 'large':
                # Large datasets: complex models can handle the data
                models = [
                    {'name': 'XGBoost', 'category': 'Tree-based models'},
                    {'name': 'RandomForest', 'category': 'Tree-based models'},
                    {'name': 'MLPClassifier', 'category': 'Neural networks'},
                    {'name': 'VotingClassifier', 'category': 'Ensemble methods'},
                    {'name': 'SVC', 'category': 'SVM and kernel methods'}
                ]
            else:  # medium
                models = [
                    {'name': 'RandomForest', 'category': 'Tree-based models'},
                    {'name': 'XGBoost', 'category': 'Tree-based models'},
                    {'name': 'LogisticRegression', 'category': 'Linear models'},
                    {'name': 'SVC', 'category': 'SVM and kernel methods'},
                    {'name': 'MLPClassifier', 'category': 'Neural networks'}
                ]
        else:  # regression
            if size_category == 'small':
                models = [
                    {'name': 'LinearRegression', 'category': 'Linear models'},
                    {'name': 'Ridge', 'category': 'Linear models'},
                    {'name': 'RandomForest', 'category': 'Tree-based models'},
                    {'name': 'MLPRegressor', 'category': 'Neural networks'},
                    {'name': 'SVR', 'category': 'SVM and kernel methods'}
                ]
            elif size_category == 'large':
                models = [
                    {'name': 'XGBoost', 'category': 'Tree-based models'},
                    {'name': 'RandomForest', 'category': 'Tree-based models'},
                    {'name': 'MLPRegressor', 'category': 'Neural networks'},
                    {'name': 'GradientBoosting', 'category': 'Tree-based models'},
                    {'name': 'SVR', 'category': 'SVM and kernel methods'}
                ]
            else:  # medium
                models = [
                    {'name': 'RandomForest', 'category': 'Tree-based models'},
                    {'name': 'XGBoost', 'category': 'Tree-based models'},
                    {'name': 'LinearRegression', 'category': 'Linear models'},
                    {'name': 'MLPRegressor', 'category': 'Neural networks'},
                    {'name': 'SVR', 'category': 'SVM and kernel methods'}
                ]
        
        return models
    
    def generate_performance_expectations(self, characteristics: Dict[str, Any], model_name: str) -> str:
        """Generate performance expectations for a specific model"""
        
        problem_type = characteristics.get('problem_type', 'classification')
        size_category = characteristics.get('size_category', 'medium')
        is_balanced = characteristics.get('is_balanced', True)
        
        expectations = []
        
        if problem_type == 'classification':
            if is_balanced:
                expectations.append("Target: Accuracy > 0.85, F1-score > 0.80")
            else:
                expectations.append("Target: F1-score > 0.75, Precision > 0.70, Recall > 0.70")
            
            if size_category == 'small':
                expectations.append("Focus on preventing overfitting with regularization")
            elif size_category == 'large':
                expectations.append("Leverage large dataset for complex model training")
        else:  # regression
            expectations.append("Target: R² > 0.80, RMSE < 0.15 * target_std")
            
            if size_category == 'small':
                expectations.append("Use simple models to avoid overfitting")
            elif size_category == 'large':
                expectations.append("Complex models can capture non-linear patterns")
        
        # Model-specific expectations
        if 'RandomForest' in model_name or 'XGBoost' in model_name:
            expectations.append("Tree-based models: Good for non-linear relationships and feature importance")
        elif 'Logistic' in model_name or 'Linear' in model_name:
            expectations.append("Linear models: Fast training, good interpretability, may need feature engineering")
        elif 'SVC' in model_name or 'SVR' in model_name:
            expectations.append("SVM: Good for high-dimensional data, kernel selection is crucial")
        elif 'MLP' in model_name:
            expectations.append("Neural networks: Can capture complex patterns, requires careful hyperparameter tuning")
        
        return "; ".join(expectations)
    
    async def create_sub_training_agents(self, selected_models: List[Dict[str, str]], 
                                       characteristics: Dict[str, Any]) -> List[Any]:
        """Create Sub Training Agents for the selected models"""
        
        # Import locally to avoid circular dependency
        from services.agent_service import create_sub_training_agent
        
        sub_agents = []
        
        for model_info in selected_models:
            model_name = model_info['name']
            model_category = model_info['category']
            performance_expectations = self.generate_performance_expectations(characteristics, model_name)
            
            agent = create_sub_training_agent(
                model_name=model_name,
                model_category=model_category,
                dataset_characteristics=characteristics,
                performance_expectations=performance_expectations
            )
            
            sub_agents.append({
                'agent': agent,
                'model_name': model_name,
                'model_category': model_category
            })
        
        return sub_agents
    
    async def execute_parallel_training(self, sub_agents: List[Dict], dataset_path: str, 
                                      target_column: str, ctx: SimpleNamespace) -> List[Dict[str, Any]]:
        """Execute training for all Sub Training Agents in parallel"""
        
        async def train_single_agent(agent_info: Dict) -> Dict[str, Any]:
            """Train a single Sub Training Agent"""
            try:
                agent = agent_info['agent']
                model_name = agent_info['model_name']
                
                training_request = f"""
Train {model_name} model on the preprocessed dataset.

Dataset: {dataset_path}
Target Column: {target_column}

Execute the complete training workflow:
1. Load preprocessed dataset
2. Implement {model_name} with initial hyperparameters
3. Perform cross-validation
4. Apply hyperparameter tuning
5. Train final model
6. Generate performance metrics
7. Save model artifacts

Return results in the specified JSON format.
"""
                
                result = await Runner.run(agent, training_request, context=ctx, max_turns=20)
                
                return {
                    'model_name': model_name,
                    'status': 'completed',
                    'result': result.final_output if hasattr(result, 'final_output') else str(result),
                    'agent': agent_info
                }
                
            except Exception as e:
                return {
                    'model_name': agent_info['model_name'],
                    'status': 'failed',
                    'error': str(e),
                    'agent': agent_info
                }
        
        # Execute all training tasks in parallel
        training_tasks = [train_single_agent(agent_info) for agent_info in sub_agents]
        results = await asyncio.gather(*training_tasks, return_exceptions=True)
        
        return results
    
    async def evaluate_results(self, training_results: List[Dict[str, Any]], 
                             characteristics: Dict[str, Any], ctx: SimpleNamespace) -> Dict[str, Any]:
        """Send results to Evaluator Agent for analysis"""
        
        # Prepare evaluation request
        evaluation_request = f"""
Analyze the results from {len(training_results)} trained models.

Dataset Characteristics:
- Problem Type: {characteristics.get('problem_type', 'Unknown')}
- Size: {characteristics.get('size', 'Unknown')} samples
- Target Column: {characteristics.get('target_column', 'Unknown')}
- Balanced: {characteristics.get('is_balanced', 'Unknown')}

Training Results:
{json.dumps(training_results, indent=2)}

Please:
1. Establish the most important metric based on dataset context
2. Rank all models by performance
3. Identify strengths and weaknesses
4. Generate model-specific tuning recommendations
5. Assess convergence potential

Return analysis in the specified JSON format.
"""
        
        result = await Runner.run(self.evaluator_agent, evaluation_request, context=ctx, max_turns=10)
        
        return {
            'evaluation_result': result.final_output if hasattr(result, 'final_output') else str(result),
            'training_results': training_results
        }
    
    async def run_training_pipeline(self, dataset_path: str, target_column: str, 
                                  ctx: SimpleNamespace, max_iterations: int = 3) -> Dict[str, Any]:
        """Run the complete training pipeline with iterative improvement"""
        
        print(f"Starting training pipeline for dataset: {dataset_path}")
        
        # Step 1: Analyze dataset characteristics
        print("Step 1: Analyzing dataset characteristics...")
        characteristics = await self.analyze_dataset_characteristics(dataset_path, target_column, ctx)
        print(f"Dataset characteristics: {characteristics}")
        
        # Step 2: Select top 5 models
        print("Step 2: Selecting top 5 models...")
        selected_models = self.select_top_5_models(characteristics)
        print(f"Selected models: {[m['name'] for m in selected_models]}")
        
        # Step 3: Create Sub Training Agents
        print("Step 3: Creating Sub Training Agents...")
        sub_agents = await self.create_sub_training_agents(selected_models, characteristics)
        
        # Step 4: Execute parallel training
        print("Step 4: Executing parallel training...")
        training_results = await self.execute_parallel_training(sub_agents, dataset_path, target_column, ctx)
        
        # Step 5: Evaluate results
        print("Step 5: Evaluating results...")
        evaluation_results = await self.evaluate_results(training_results, characteristics, ctx)
        
        # Step 6: Iterative improvement (if needed)
        iteration = 1
        while iteration < max_iterations:
            print(f"Iteration {iteration + 1}: Checking for improvement opportunities...")
            
            # Parse evaluation results to check if improvement is needed
            try:
                eval_data = json.loads(evaluation_results['evaluation_result'])
                convergence_assessment = eval_data.get('convergence_assessment', '')
                
                if 'converged' in convergence_assessment.lower() or 'satisfactory' in convergence_assessment.lower():
                    print("Convergence achieved. Stopping iterations.")
                    break
                
                # If improvement needed, run another iteration
                print("Improvement opportunities identified. Running another iteration...")
                training_results = await self.execute_parallel_training(sub_agents, dataset_path, target_column, ctx)
                evaluation_results = await self.evaluate_results(training_results, characteristics, ctx)
                iteration += 1
                
            except Exception as e:
                print(f"Error in iterative improvement: {e}")
                break
        
        return {
            'characteristics': characteristics,
            'selected_models': selected_models,
            'final_results': evaluation_results,
            'iterations_completed': iteration,
            'status': 'completed'
        }
