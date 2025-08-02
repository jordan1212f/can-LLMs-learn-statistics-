"""
RAG System Testing and Experimentation Framework
===============================================

This script demonstrates how to test and run experiments on your RAG system
to evaluate performance, optimize parameters, and ensure quality for educational content.

Author: Jordan Fernandes
Date: July 19, 2025
Purpose: Dissertation research - RAG system evaluation and optimization
"""

import json
import time
import numpy as np
from typing import List, Dict, Any, Tuple
from datetime import datetime
from pathlib import Path
import logging

# Optional imports for visualization (install with pip if needed)
try:
    import pandas as pd
    import matplotlib.pyplot as plt
    import seaborn as sns
    VISUALIZATION_AVAILABLE = True
except ImportError:
    VISUALIZATION_AVAILABLE = False
    print("Note: pandas, matplotlib, seaborn not available. Install with: pip install pandas matplotlib seaborn")

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class RAGSystemTester:
    """
    Comprehensive testing framework for RAG systems focused on educational content.
    """
    
    def __init__(self, rag_system, output_dir: str = "evaluation_results"):
        """
        Initialize the RAG testing framework.
        
        Args:
            rag_system: Your initialized MultiDocumentRAGSystem instance
            output_dir: Directory to save test results
        """
        self.rag_system = rag_system
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
        
        # Test results storage
        self.test_results = []
        self.parameter_experiments = []
        self.evaluation_metrics = {}
        
        logger.info(f"RAG System Tester initialized. Results will be saved to: {self.output_dir}")
    
    def create_test_questions(self) -> List[Dict[str, Any]]:
        """
        Create a comprehensive set of test questions for statistics education.
        
        Returns:
            List of test questions with expected answer types and difficulty levels
        """
        test_questions = [
            # Basic Concepts
            {
                "question": "What is the Central Limit Theorem?",
                "category": "fundamental_concepts",
                "difficulty": "basic",
                "expected_topics": ["central limit theorem", "sampling distribution", "normal distribution"],
                "answer_type": "definition_explanation"
            },
            {
                "question": "What is a p-value and how do you interpret it?",
                "category": "hypothesis_testing",
                "difficulty": "intermediate",
                "expected_topics": ["p-value", "hypothesis testing", "significance"],
                "answer_type": "definition_interpretation"
            },
            {
                "question": "How do you calculate a 95% confidence interval for a population mean?",
                "category": "confidence_intervals",
                "difficulty": "intermediate",
                "expected_topics": ["confidence interval", "margin of error", "critical value"],
                "answer_type": "procedure_calculation"
            },
            
            # Advanced Concepts
            {
                "question": "What is the difference between Type I and Type II errors?",
                "category": "hypothesis_testing",
                "difficulty": "intermediate",
                "expected_topics": ["type I error", "type II error", "hypothesis testing"],
                "answer_type": "comparison_explanation"
            },
            {
                "question": "When should you use a t-distribution instead of a normal distribution?",
                "category": "distributions",
                "difficulty": "advanced",
                "expected_topics": ["t-distribution", "degrees of freedom", "sample size"],
                "answer_type": "conditional_procedure"
            },
            {
                "question": "Explain the assumptions for ANOVA and what happens if they're violated",
                "category": "anova",
                "difficulty": "advanced",
                "expected_topics": ["ANOVA", "assumptions", "independence", "normality"],
                "answer_type": "assumptions_consequences"
            },
            
            # Application Questions
            {
                "question": "A researcher wants to test if a new teaching method improves test scores. Design a hypothesis test.",
                "category": "hypothesis_testing",
                "difficulty": "advanced",
                "expected_topics": ["hypothesis test design", "null hypothesis", "alternative hypothesis"],
                "answer_type": "problem_solving"
            },
            {
                "question": "Interpret this result: p-value = 0.03, α = 0.05",
                "category": "hypothesis_testing",
                "difficulty": "basic",
                "expected_topics": ["p-value", "significance level", "decision"],
                "answer_type": "interpretation"
            },
            
            # Computational Questions
            {
                "question": "What factors affect the width of a confidence interval?",
                "category": "confidence_intervals",
                "difficulty": "intermediate",
                "expected_topics": ["confidence level", "sample size", "standard deviation"],
                "answer_type": "factor_analysis"
            },
            {
                "question": "How do you check if data meets the normality assumption?",
                "category": "assumptions",
                "difficulty": "intermediate",
                "expected_topics": ["normality", "QQ plot", "shapiro-wilk", "visual inspection"],
                "answer_type": "diagnostic_procedure"
            }
        ]
        
        logger.info(f"Created {len(test_questions)} test questions across {len(set(q['category'] for q in test_questions))} categories")
        return test_questions
    
    def evaluate_answer_quality(self, question: str, answer: str, expected_topics: List[str]) -> Dict[str, Any]:
        """
        Evaluate the quality of a RAG system answer.
        
        Args:
            question: The input question
            answer: Generated answer from RAG system
            expected_topics: Topics that should be covered in a good answer
        
        Returns:
            Dictionary with quality metrics
        """
        # Convert to lowercase for comparison
        answer_lower = answer.lower()
        
        # Topic coverage analysis
        topics_covered = []
        topics_missing = []
        
        for topic in expected_topics:
            if topic.lower() in answer_lower:
                topics_covered.append(topic)
            else:
                topics_missing.append(topic)
        
        coverage_score = len(topics_covered) / len(expected_topics) if expected_topics else 0
        
        # Length and detail analysis
        answer_length = len(answer.split())
        detail_score = min(answer_length / 100, 1.0)  # Normalize to 0-1, expect ~100 words
        
        # Source citation analysis
        has_source_references = any(indicator in answer_lower for indicator in 
                                  ['source:', 'page', 'according to', 'from', 'textbook'])
        
        # Statistical terminology usage
        stats_terms = ['hypothesis', 'distribution', 'significance', 'confidence', 'probability', 
                      'sample', 'population', 'statistic', 'parameter', 'variance', 'deviation']
        stats_terms_used = sum(1 for term in stats_terms if term in answer_lower)
        terminology_score = min(stats_terms_used / 5, 1.0)  # Expect at least 5 terms
        
        # Overall quality score (weighted average)
        quality_score = (
            coverage_score * 0.4 +      # 40% - topic coverage
            detail_score * 0.2 +        # 20% - answer detail
            terminology_score * 0.2 +   # 20% - proper terminology
            (1.0 if has_source_references else 0.0) * 0.2  # 20% - source citations
        )
        
        return {
            'coverage_score': coverage_score,
            'topics_covered': topics_covered,
            'topics_missing': topics_missing,
            'answer_length': answer_length,
            'detail_score': detail_score,
            'has_source_references': has_source_references,
            'terminology_score': terminology_score,
            'stats_terms_used': stats_terms_used,
            'quality_score': quality_score
        }
    
    def run_comprehensive_test(self, test_questions: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Run comprehensive testing of the RAG system.
        
        Args:
            test_questions: List of test question dictionaries
        
        Returns:
            Complete test results
        """
        logger.info(f"Starting comprehensive test with {len(test_questions)} questions...")
        
        results = {
            'test_metadata': {
                'start_time': datetime.now().isoformat(),
                'total_questions': len(test_questions),
                'rag_system_config': {
                    'chunk_size': getattr(self.rag_system, 'chunk_size', 'unknown'),
                    'chunk_overlap': getattr(self.rag_system, 'chunk_overlap', 'unknown'),
                    'embedding_model': getattr(self.rag_system, 'embedding_model', 'unknown'),
                    'llm_model': getattr(self.rag_system, 'llm_model', 'unknown')
                }
            },
            'question_results': [],
            'aggregate_metrics': {}
        }
        
        total_time = 0
        category_scores = {}
        difficulty_scores = {}
        
        for i, test_q in enumerate(test_questions, 1):
            logger.info(f"Testing question {i}/{len(test_questions)}: {test_q['category']}")
            
            # Time the response
            start_time = time.time()
            try:
                answer = self.rag_system.ask_question(test_q['question'], verbose=False)
                response_time = time.time() - start_time
                total_time += response_time
                
                # Evaluate answer quality
                quality_metrics = self.evaluate_answer_quality(
                    test_q['question'], 
                    answer, 
                    test_q['expected_topics']
                )
                
                # Store results
                question_result = {
                    'question_id': i,
                    'question': test_q['question'],
                    'category': test_q['category'],
                    'difficulty': test_q['difficulty'],
                    'expected_topics': test_q['expected_topics'],
                    'answer': answer,
                    'response_time': response_time,
                    'quality_metrics': quality_metrics
                }
                
                results['question_results'].append(question_result)
                
                # Aggregate by category
                if test_q['category'] not in category_scores:
                    category_scores[test_q['category']] = []
                category_scores[test_q['category']].append(quality_metrics['quality_score'])
                
                # Aggregate by difficulty
                if test_q['difficulty'] not in difficulty_scores:
                    difficulty_scores[test_q['difficulty']] = []
                difficulty_scores[test_q['difficulty']].append(quality_metrics['quality_score'])
                
            except Exception as e:
                logger.error(f"Error processing question {i}: {e}")
                question_result = {
                    'question_id': i,
                    'question': test_q['question'],
                    'category': test_q['category'],
                    'difficulty': test_q['difficulty'],
                    'error': str(e),
                    'response_time': None,
                    'quality_metrics': None
                }
                results['question_results'].append(question_result)
        
        # Calculate aggregate metrics
        valid_results = [r for r in results['question_results'] if 'error' not in r]
        if valid_results:
            quality_scores = [r['quality_metrics']['quality_score'] for r in valid_results]
            response_times = [r['response_time'] for r in valid_results]
            
            results['aggregate_metrics'] = {
                'overall_quality_score': sum(quality_scores) / len(quality_scores),
                'average_response_time': sum(response_times) / len(response_times),
                'total_test_time': total_time,
                'success_rate': len(valid_results) / len(test_questions),
                'category_scores': {cat: sum(scores)/len(scores) for cat, scores in category_scores.items()},
                'difficulty_scores': {diff: sum(scores)/len(scores) for diff, scores in difficulty_scores.items()}
            }
        
        results['test_metadata']['end_time'] = datetime.now().isoformat()
        results['test_metadata']['total_duration'] = total_time
        
        logger.info(f"Comprehensive test completed. Overall quality score: {results['aggregate_metrics'].get('overall_quality_score', 'N/A'):.3f}")
        
        return results
    
    def parameter_experiment(self, parameter_configs: List[Dict[str, Any]], test_questions: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Run experiments with different RAG parameters.
        
        Args:
            parameter_configs: List of parameter configurations to test
            test_questions: Subset of test questions for parameter testing
        
        Returns:
            Parameter experiment results
        """
        logger.info(f"Starting parameter experiments with {len(parameter_configs)} configurations...")
        
        experiment_results = {
            'experiment_metadata': {
                'start_time': datetime.now().isoformat(),
                'configurations_tested': len(parameter_configs),
                'test_questions_used': len(test_questions)
            },
            'configuration_results': []
        }
        
        for i, config in enumerate(parameter_configs, 1):
            logger.info(f"Testing configuration {i}/{len(parameter_configs)}: {config}")
            
            try:
                # Note: In practice, you'd need to recreate the RAG system with new parameters
                # This is a simplified version that assumes parameter changes are possible
                
                # For demonstration, we'll simulate different retrieval parameters
                original_k = self.rag_system.retriever.search_kwargs.get('k', 6)
                
                if 'retrieval_k' in config:
                    self.rag_system.retriever.search_kwargs['k'] = config['retrieval_k']
                
                # Run a subset of test questions
                subset_results = []
                for test_q in test_questions[:5]:  # Use first 5 questions for parameter testing
                    start_time = time.time()
                    answer = self.rag_system.ask_question(test_q['question'], verbose=False)
                    response_time = time.time() - start_time
                    
                    quality_metrics = self.evaluate_answer_quality(
                        test_q['question'], answer, test_q['expected_topics']
                    )
                    
                    subset_results.append({
                        'question': test_q['question'],
                        'quality_score': quality_metrics['quality_score'],
                        'response_time': response_time
                    })
                
                # Calculate average metrics for this configuration
                avg_quality = sum(r['quality_score'] for r in subset_results) / len(subset_results)
                avg_response_time = sum(r['response_time'] for r in subset_results) / len(subset_results)
                
                config_result = {
                    'configuration': config,
                    'average_quality_score': avg_quality,
                    'average_response_time': avg_response_time,
                    'detailed_results': subset_results
                }
                
                experiment_results['configuration_results'].append(config_result)
                
                # Restore original parameter
                if 'retrieval_k' in config:
                    self.rag_system.retriever.search_kwargs['k'] = original_k
                
                logger.info(f"Configuration {i} completed. Quality: {avg_quality:.3f}, Time: {avg_response_time:.2f}s")
                
            except Exception as e:
                logger.error(f"Error testing configuration {i}: {e}")
                experiment_results['configuration_results'].append({
                    'configuration': config,
                    'error': str(e)
                })
        
        experiment_results['experiment_metadata']['end_time'] = datetime.now().isoformat()
        
        # Find best configuration
        valid_configs = [r for r in experiment_results['configuration_results'] if 'error' not in r]
        if valid_configs:
            best_config = max(valid_configs, key=lambda x: x['average_quality_score'])
            experiment_results['best_configuration'] = best_config
            logger.info(f"Best configuration: {best_config['configuration']} (Quality: {best_config['average_quality_score']:.3f})")
        
        return experiment_results
    
    def visualize_results(self, test_results: Dict[str, Any]):
        """
        Create visualizations of test results.
        
        Args:
            test_results: Results from comprehensive testing
        """
        if not VISUALIZATION_AVAILABLE:
            logger.warning("Visualization libraries not available. Install with: pip install pandas matplotlib seaborn numpy")
            return
            
        logger.info("Creating result visualizations...")
        
        # Extract data for plotting
        valid_results = [r for r in test_results['question_results'] if 'error' not in r]
        
        if not valid_results:
            logger.warning("No valid results to visualize")
            return
        
        # Prepare data
        categories = [r['category'] for r in valid_results]
        difficulties = [r['difficulty'] for r in valid_results]
        quality_scores = [r['quality_metrics']['quality_score'] for r in valid_results]
        response_times = [r['response_time'] for r in valid_results]
        
        # Create subplots
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 12))
        fig.suptitle('RAG System Performance Analysis', fontsize=16, fontweight='bold')
        
        # 1. Quality scores by category
        category_df = pd.DataFrame({'Category': categories, 'Quality Score': quality_scores})
        category_avg = category_df.groupby('Category')['Quality Score'].mean().sort_values(ascending=False)
        
        ax1.bar(range(len(category_avg)), category_avg.values, color='skyblue')
        ax1.set_xlabel('Question Category')
        ax1.set_ylabel('Average Quality Score')
        ax1.set_title('Performance by Question Category')
        ax1.set_xticks(range(len(category_avg)))
        ax1.set_xticklabels(category_avg.index, rotation=45, ha='right')
        ax1.grid(axis='y', alpha=0.3)
        
        # 2. Quality scores by difficulty
        difficulty_df = pd.DataFrame({'Difficulty': difficulties, 'Quality Score': quality_scores})
        difficulty_avg = difficulty_df.groupby('Difficulty')['Quality Score'].mean()
        
        ax2.bar(difficulty_avg.index, difficulty_avg.values, color='lightcoral')
        ax2.set_xlabel('Difficulty Level')
        ax2.set_ylabel('Average Quality Score')
        ax2.set_title('Performance by Difficulty Level')
        ax2.grid(axis='y', alpha=0.3)
        
        # 3. Response time distribution
        ax3.hist(response_times, bins=10, color='lightgreen', alpha=0.7, edgecolor='black')
        ax3.set_xlabel('Response Time (seconds)')
        ax3.set_ylabel('Frequency')
        ax3.set_title('Response Time Distribution')
        ax3.axvline(sum(response_times)/len(response_times), color='red', linestyle='--', 
                   label=f'Average: {sum(response_times)/len(response_times):.2f}s')
        ax3.legend()
        ax3.grid(alpha=0.3)
        
        # 4. Quality vs Response Time scatter
        ax4.scatter(response_times, quality_scores, alpha=0.6, color='purple')
        ax4.set_xlabel('Response Time (seconds)')
        ax4.set_ylabel('Quality Score')
        ax4.set_title('Quality vs Response Time')
        ax4.grid(alpha=0.3)
        
        # Add correlation line if numpy is available
        try:
            z = np.polyfit(response_times, quality_scores, 1)
            p = np.poly1d(z)
            ax4.plot(sorted(response_times), p(sorted(response_times)), "r--", alpha=0.8, 
                    label=f'Correlation trend')
            ax4.legend()
        except:
            pass
        
        plt.tight_layout()
        
        # Save plot
        plot_path = self.output_dir / f"rag_performance_analysis_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png"
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        logger.info(f"Visualization saved to: {plot_path}")
        
        plt.show()
    
    def save_results(self, test_results: Dict[str, Any], filename: str = None):
        """
        Save test results to JSON file.
        
        Args:
            test_results: Results to save
            filename: Optional filename, otherwise auto-generated
        """
        if filename is None:
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            filename = f"rag_test_results_{timestamp}.json"
        
        filepath = self.output_dir / filename
        
        with open(filepath, 'w') as f:
            json.dump(test_results, f, indent=2, ensure_ascii=False)
        
        logger.info(f"Test results saved to: {filepath}")
        return filepath
    
    def generate_report(self, test_results: Dict[str, Any]) -> str:
        """
        Generate a comprehensive test report.
        
        Args:
            test_results: Results from comprehensive testing
        
        Returns:
            Formatted report string
        """
        report = []
        report.append("="*80)
        report.append("RAG SYSTEM PERFORMANCE REPORT")
        report.append("="*80)
        
        # Basic metrics
        metrics = test_results.get('aggregate_metrics', {})
        report.append(f"\n📊 OVERALL PERFORMANCE:")
        report.append(f"  Overall Quality Score: {metrics.get('overall_quality_score', 'N/A'):.3f}")
        report.append(f"  Success Rate: {metrics.get('success_rate', 'N/A'):.1%}")
        report.append(f"  Average Response Time: {metrics.get('average_response_time', 'N/A'):.2f} seconds")
        report.append(f"  Total Test Duration: {metrics.get('total_test_time', 'N/A'):.2f} seconds")
        
        # Category performance
        category_scores = metrics.get('category_scores', {})
        if category_scores:
            report.append(f"\n📚 PERFORMANCE BY CATEGORY:")
            for category, score in sorted(category_scores.items(), key=lambda x: x[1], reverse=True):
                report.append(f"  {category}: {score:.3f}")
        
        # Difficulty performance
        difficulty_scores = metrics.get('difficulty_scores', {})
        if difficulty_scores:
            report.append(f"\n🎯 PERFORMANCE BY DIFFICULTY:")
            for difficulty, score in sorted(difficulty_scores.items(), key=lambda x: x[1], reverse=True):
                report.append(f"  {difficulty}: {score:.3f}")
        
        # Best and worst performing questions
        valid_results = [r for r in test_results.get('question_results', []) if 'error' not in r]
        if valid_results:
            sorted_results = sorted(valid_results, key=lambda x: x['quality_metrics']['quality_score'], reverse=True)
            
            report.append(f"\n🏆 BEST PERFORMING QUESTION:")
            best = sorted_results[0]
            report.append(f"  Question: {best['question']}")
            report.append(f"  Quality Score: {best['quality_metrics']['quality_score']:.3f}")
            report.append(f"  Category: {best['category']}")
            
            report.append(f"\n⚠️ LOWEST PERFORMING QUESTION:")
            worst = sorted_results[-1]
            report.append(f"  Question: {worst['question']}")
            report.append(f"  Quality Score: {worst['quality_metrics']['quality_score']:.3f}")
            report.append(f"  Category: {worst['category']}")
            report.append(f"  Missing Topics: {worst['quality_metrics']['topics_missing']}")
        
        # Recommendations
        report.append(f"\n💡 RECOMMENDATIONS:")
        if metrics.get('overall_quality_score', 0) < 0.7:
            report.append(f"  - Consider improving document coverage or chunk size")
            report.append(f"  - Review prompt engineering for better responses")
        
        if metrics.get('average_response_time', 0) > 10:
            report.append(f"  - Response times are high, consider optimizing retrieval")
            
        if category_scores:
            worst_category = min(category_scores.items(), key=lambda x: x[1])
            if worst_category[1] < 0.6:
                report.append(f"  - Focus on improving {worst_category[0]} category (score: {worst_category[1]:.3f})")
        
        report.append("\n" + "="*80)
        
        return "\n".join(report)

# Example usage and testing functions
def run_example_experiment():
    """
    Example of how to run comprehensive RAG system testing.
    """
    # This would be used with your actual RAG system
    print("""
    EXAMPLE: Running RAG System Testing and Experimentation
    
    # 1. Initialize your RAG system (from previous script)
    from multi_document_rag_system import MultiDocumentRAGSystem
    
    rag_system = MultiDocumentRAGSystem(...)
    # ... setup your system ...
    
    # 2. Initialize tester
    tester = RAGSystemTester(rag_system)
    
    # 3. Create test questions
    test_questions = tester.create_test_questions()
    
    # 4. Run comprehensive test
    results = tester.run_comprehensive_test(test_questions)
    
    # 5. Run parameter experiments
    param_configs = [
        {'retrieval_k': 4},
        {'retrieval_k': 6},
        {'retrieval_k': 8},
        {'retrieval_k': 10}
    ]
    param_results = tester.parameter_experiment(param_configs, test_questions[:5])
    
    # 6. Visualize and save results
    tester.visualize_results(results)
    tester.save_results(results)
    
    # 7. Generate report
    report = tester.generate_report(results)
    print(report)
    """)

if __name__ == "__main__":
    run_example_experiment()
