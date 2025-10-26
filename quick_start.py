"""
Quick Start Script for HeartGuard AI
Simple commands to get the project running
"""

import os
import sys
import subprocess

def quick_start():
    """Quick start the project"""
    print("🚀 HeartGuard AI - Quick Start")
    print("=" * 40)
    
    print("\n📋 Available Commands:")
    print("1. Install dependencies")
    print("2. Train ML models")
    print("3. Start Flask frontend")
    print("4. Test API connection")
    print("5. Run complete setup")
    print("0. Exit")
    
    while True:
        choice = input("\n🤔 What would you like to do? (0-5): ").strip()
        
        if choice == "0":
            print("👋 Goodbye!")
            break
        elif choice == "1":
            print("\n📦 Installing dependencies...")
            subprocess.run([sys.executable, "-m", "pip", "install", "-r", "requirements.txt"])
            subprocess.run([sys.executable, "-m", "pip", "install", "-r", "flask_requirements.txt"])
            print("✅ Dependencies installed!")
        elif choice == "2":
            print("\n🤖 Training ML models...")
            subprocess.run([sys.executable, "app/main.py"])
            print("✅ Models trained!")
        elif choice == "3":
            print("\n🌐 Starting Flask frontend...")
            print("Open: http://localhost:5000")
            subprocess.run([sys.executable, "run_flask.py"])
        elif choice == "4":
            print("\n🧪 Testing API connection...")
            subprocess.run([sys.executable, "test_prediction.py"])
        elif choice == "5":
            print("\n🚀 Running complete setup...")
            subprocess.run([sys.executable, "run_complete_project.py"])
        else:
            print("❌ Invalid choice. Please try again.")

if __name__ == "__main__":
    quick_start()
