#!/usr/bin/env python3
"""
Data Science Internship Project - Main Runner Script

This script provides a unified interface to run all four data science tasks
in the internship portfolio.

Author: Parth Parmar
Date: June 2025
"""

import os
import sys
import subprocess
import argparse
from pathlib import Path

class DataScienceProjectRunner:
    """
    Main runner class for the Data Science Internship Portfolio
    """
    
    def __init__(self):
        self.project_root = Path(__file__).parent
        self.tasks = {
            'iris': {
                'path': 'task1-iris',
                'script': 'task1.py',
                'description': 'Iris Flower Classification using Multiple ML Algorithms'
            },
            'unemployment': {
                'path': 'task2-unemployment', 
                'script': 'task2.py',
                'description': 'Unemployment Analysis in India with COVID-19 Impact'
            },
            'carprice': {
                'path': 'task3-carprice',
                'script': 'task3.py', 
                'description': 'Car Price Prediction using Regression Models'
            },
            'sales': {
                'path': 'task4-sales',
                'script': 'task4.py',
                'description': 'Sales Prediction and Marketing Analytics'
            }
        }
    
    def install_requirements(self, task_name):
        """Install requirements for a specific task"""
        task_path = self.project_root / self.tasks[task_name]['path']
        requirements_file = task_path / 'requirements.txt'
        
        if requirements_file.exists():
            print(f"Installing requirements for {task_name}...")
            try:
                subprocess.run([
                    sys.executable, '-m', 'pip', 'install', '-r', str(requirements_file)
                ], check=True, cwd=task_path)
                print(f"✅ Requirements installed successfully for {task_name}")
                return True
            except subprocess.CalledProcessError as e:
                print(f"❌ Error installing requirements for {task_name}: {e}")
                return False
        else:
            print(f"⚠️ No requirements.txt found for {task_name}")
            return True
    
    def run_task(self, task_name):
        """Run a specific task"""
        if task_name not in self.tasks:
            print(f"❌ Task '{task_name}' not found. Available tasks: {list(self.tasks.keys())}")
            return False
        
        task_info = self.tasks[task_name]
        task_path = self.project_root / task_info['path']
        script_path = task_path / task_info['script']
        
        if not script_path.exists():
            print(f"❌ Script not found: {script_path}")
            return False
        
        print(f"\n🚀 Running Task: {task_info['description']}")
        print(f"📁 Working Directory: {task_path}")
        print(f"🐍 Script: {task_info['script']}")
        print("-" * 60)
        
        try:
            # Install requirements first
            if not self.install_requirements(task_name):
                return False
            
            # Run the task
            result = subprocess.run([
                sys.executable, script_path
            ], cwd=task_path, capture_output=False)
            
            if result.returncode == 0:
                print(f"✅ Task '{task_name}' completed successfully!")
                return True
            else:
                print(f"❌ Task '{task_name}' failed with exit code {result.returncode}")
                return False
                
        except Exception as e:
            print(f"❌ Error running task '{task_name}': {e}")
            return False
    
    def run_all_tasks(self):
        """Run all tasks in sequence"""
        print("🔬 Data Science Internship Portfolio - Running All Tasks")
        print("=" * 60)
        
        results = {}
        for task_name, task_info in self.tasks.items():
            print(f"\n📊 Starting Task {task_name.upper()}: {task_info['description']}")
            results[task_name] = self.run_task(task_name)
            
            if results[task_name]:
                print(f"✅ Task {task_name} completed successfully")
            else:
                print(f"❌ Task {task_name} failed")
            
            print("-" * 60)
        
        # Summary
        print("\n📋 EXECUTION SUMMARY")
        print("=" * 40)
        successful_tasks = [task for task, success in results.items() if success]
        failed_tasks = [task for task, success in results.items() if not success]
        
        print(f"✅ Successful Tasks ({len(successful_tasks)}): {', '.join(successful_tasks)}")
        if failed_tasks:
            print(f"❌ Failed Tasks ({len(failed_tasks)}): {', '.join(failed_tasks)}")
        
        success_rate = len(successful_tasks) / len(self.tasks) * 100
        print(f"📊 Success Rate: {success_rate:.1f}%")
        
        return len(failed_tasks) == 0
    
    def list_tasks(self):
        """List all available tasks"""
        print("📋 Available Data Science Tasks:")
        print("=" * 50)
        for task_name, task_info in self.tasks.items():
            print(f"🔹 {task_name}: {task_info['description']}")
            print(f"   📁 Path: {task_info['path']}")
            print(f"   🐍 Script: {task_info['script']}")
            print()
    
    def check_environment(self):
        """Check if the environment is properly set up"""
        print("🔍 Checking Environment Setup...")
        print("-" * 40)
        
        # Check Python version
        python_version = sys.version_info
        print(f"🐍 Python Version: {python_version.major}.{python_version.minor}.{python_version.micro}")
        
        if python_version.major < 3 or (python_version.major == 3 and python_version.minor < 7):
            print("⚠️ Warning: Python 3.7+ is recommended")
        
        # Check if all task directories exist
        missing_tasks = []
        for task_name, task_info in self.tasks.items():
            task_path = self.project_root / task_info['path']
            script_path = task_path / task_info['script']
            
            if not task_path.exists():
                missing_tasks.append(f"{task_name} (directory missing)")
            elif not script_path.exists():
                missing_tasks.append(f"{task_name} (script missing)")
            else:
                print(f"✅ {task_name}: OK")
        
        if missing_tasks:
            print(f"❌ Missing components: {', '.join(missing_tasks)}")
            return False
        
        print("✅ Environment check passed!")
        return True

def main():
    """Main function to handle command line arguments"""
    parser = argparse.ArgumentParser(
        description="Data Science Internship Project Runner",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python run_portfolio.py --list              # List all available tasks
  python run_portfolio.py --check             # Check environment setup
  python run_portfolio.py --task iris         # Run only iris classification
  python run_portfolio.py --all               # Run all tasks
  python run_portfolio.py --task unemployment # Run unemployment analysis
        """
    )
    
    parser.add_argument('--task', type=str, 
                       choices=['iris', 'unemployment', 'carprice', 'sales'],
                       help='Run a specific task')
    parser.add_argument('--all', action='store_true',
                       help='Run all tasks in sequence')
    parser.add_argument('--list', action='store_true',
                       help='List all available tasks')
    parser.add_argument('--check', action='store_true', 
                       help='Check environment setup')
    
    args = parser.parse_args()
    
    runner = DataScienceProjectRunner()
    
    if args.list:
        runner.list_tasks()
    elif args.check:
        runner.check_environment()
    elif args.task:
        if not runner.check_environment():
            sys.exit(1)
        success = runner.run_task(args.task)
        sys.exit(0 if success else 1)
    elif args.all:
        if not runner.check_environment():
            sys.exit(1)
        success = runner.run_all_tasks()
        sys.exit(0 if success else 1)
    else:
        parser.print_help()

if __name__ == "__main__":
    main()
