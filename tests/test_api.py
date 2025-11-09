"""
Tests for FastAPI Server
"""
import pytest
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))


class TestAPISetup:
    """Test API setup and configuration"""
    
    def test_api_file_exists(self):
        """Test API file exists"""
        api_file = Path(__file__).parent.parent / "python_scripts" / "api" / "run_api.py"
        assert api_file.exists()
    
    def test_api_imports(self):
        """Test API dependencies can be imported"""
        try:
            from fastapi import FastAPI
            from pydantic import BaseModel
            assert True
        except ImportError as e:
            pytest.skip(f"FastAPI not installed: {e}")
    
    def test_api_syntax_valid(self):
        """Test API file has valid Python syntax"""
        api_file = Path(__file__).parent.parent / "python_scripts" / "api" / "run_api.py"
        
        with open(api_file, 'r', encoding='utf-8') as f:
            code = f.read()
        
        try:
            compile(code, api_file, 'exec')
            assert True
        except SyntaxError as e:
            pytest.fail(f"Syntax error in run_api.py: {e}")


class TestDashboard:
    """Test Streamlit Dashboard"""
    
    def test_dashboard_file_exists(self):
        """Test dashboard file exists"""
        dashboard_file = (
            Path(__file__).parent.parent
            / "python_scripts"
            / "dashboard"
            / "streamlit_inference_app.py"
        )
        assert dashboard_file.exists()
    
    def test_dashboard_syntax_valid(self):
        """Test dashboard file has valid Python syntax"""
        dashboard_file = (
            Path(__file__).parent.parent
            / "python_scripts"
            / "dashboard"
            / "streamlit_inference_app.py"
        )
        
        with open(dashboard_file, 'r', encoding='utf-8') as f:
            code = f.read()
        
        try:
            compile(code, dashboard_file, 'exec')
            assert True
        except SyntaxError as e:
            pytest.fail(f"Syntax error in streamlit_inference_app.py: {e}")


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
