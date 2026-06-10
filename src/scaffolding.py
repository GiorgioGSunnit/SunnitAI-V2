"""
Scaffolding utilities for Python project development.
Provides shortcuts for common development tasks.
"""

import argparse
import subprocess
import sys
from pathlib import Path
from typing import Optional

import toml


def pni(package_name: str, version: Optional[str] = None) -> None:
    """
    Package Install: Install a dependency with pip and update pyproject.toml

    Args:
        package_name: Name of the package to install
        version: Optional version specification (e.g., ">=1.0.0")
    """
    print(f"📦 Installing package: {package_name}")

    # Construct pip install command
    install_cmd = [sys.executable, "-m", "pip", "install", package_name]

    try:
        # Install the package
        subprocess.run(install_cmd, check=True)
        print(f"✅ Successfully installed {package_name}")

        # Update pyproject.toml
        pyproject_toml_path = Path("pyproject.toml")
        if pyproject_toml_path.exists():
            with open(pyproject_toml_path, "r") as f:
                config = toml.load(f)

            # Add to dependencies
            if "project" not in config:
                config["project"] = {}
            if "dependencies" not in config["project"]:
                config["project"]["dependencies"] = []

            # Format dependency string
            dep_string = f"{package_name}{version}" if version else package_name

            # Add if not already present
            if dep_string not in config["project"]["dependencies"]:
                config["project"]["dependencies"].append(dep_string)

                # Write back to file
                with open(pyproject_toml_path, "w") as f:
                    toml.dump(config, f)

                print(f"✅ Updated pyproject.toml with {dep_string}")
            else:
                print(f"ℹ️  {dep_string} already in pyproject.toml")
        else:
            print("⚠️  pyproject.toml not found, skipping update")

    except subprocess.CalledProcessError as e:
        print(f"❌ Error installing {package_name}: {e}")
        sys.exit(1)
    except Exception as e:
        print(f"❌ Error updating pyproject.toml: {e}")


def pnd(dir_path: str, is_test: bool = False) -> None:
    """
    Package New Directory: Create a directory with __init__.py file

    Args:
        dir_path: Path to the directory to create (relative to src/ or src/test/)
        is_test: If True, create in src/test/, otherwise in src/
    """
    try:
        # Determine base directory
        if is_test:
            base_dir = Path("src/test")
        else:
            base_dir = Path("src")

        # Create full directory path
        full_dir = base_dir / dir_path
        full_dir.mkdir(parents=True, exist_ok=True)

        # Create __init__.py if it doesn't exist
        init_file = full_dir / "__init__.py"
        if not init_file.exists():
            init_file.write_text('"""Package initialization."""\n')

        print(f"✅ Created directory: {full_dir}")
        print(f"✅ Created __init__.py: {init_file}")

    except Exception as e:
        print(f"❌ Error creating directory {dir_path}: {e}")
        sys.exit(1)


def pnf(file_path: str, template: str = "basic", is_test: bool = False) -> None:
    """
    Package New File: Create a Python file with a template

    Args:
        file_path: Path to the file to create (relative to src/ or src/test/)
        template: Template type (basic, class, cli, test)
        is_test: If True, create in src/test/, otherwise in src/
    """
    try:
        # Determine base directory
        if is_test or template == "test":
            base_dir = Path("src/test")
        else:
            base_dir = Path("src")

        # Create full file path
        full_file_path = base_dir / file_path
        full_file_path.parent.mkdir(parents=True, exist_ok=True)

        # Template content
        templates = {
            "basic": '''"""
{filename}

Description: {description}
Author: SunnitAI
Date: {date}
"""
from src.registry import *

def main():
    """Main function."""
    print("Hello, World!")


if __name__ == "__main__":
    main()
''',
            "class": '''"""
{classname}

Description: {description}
Author: SunnitAI
Date: {date}
"""
from src.registry import *


class {classname}:
    """{classname} class."""
    
    def __init__(self):
        """Initialize {classname}."""
        pass
    
    def method(self):
        """Example method."""
        pass


def main():
    """Main function."""
    obj = {classname}()
    print(f"Created {classname} instance")


if __name__ == "__main__":
    main()
''',
            "cli": '''"""
{filename}

CLI tool for {description}
Author: SunnitAI
Date: {date}
"""
from src.registry import *

import argparse
import sys


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="{description}")
    parser.add_argument("--input", help="Input file")
    parser.add_argument("--output", help="Output file")
    return parser.parse_args()


def main():
    """Main function."""
    args = parse_args()
    print("CLI tool started")
    print(f"Input: {{args.input}}")
    print(f"Output: {{args.output}}")


if __name__ == "__main__":
    main()
''',
            "test": '''"""
Test module for {filename}

Description: {description}
Author: SunnitAI
Date: {date}
"""
from src.registry import *

import unittest
from unittest.mock import patch, MagicMock


class Test{testclass}(unittest.TestCase):
    """Test cases for {testclass}."""
    
    def setUp(self):
        """Set up test fixtures."""
        pass
    
    def tearDown(self):
        """Tear down test fixtures."""
        pass
    
    def test_example(self):
        """Test example."""
        self.assertTrue(True)


if __name__ == "__main__":
    unittest.main()
''',
        }

        if template not in templates:
            print(f"❌ Unknown template: {template}")
            print(f"Available templates: {list(templates.keys())}")
            sys.exit(1)

        # Get template content
        template_content = templates[template]

        # Prepare template variables
        filename = full_file_path.stem
        classname = filename.title().replace("_", "").replace("-", "")
        testclass = (
            classname.replace("Test", "") if classname.endswith("Test") else classname
        )
        description = f"{filename} module"
        date = "2024"

        # Format template
        content = template_content.format(
            filename=filename,
            classname=classname,
            testclass=testclass,
            description=description,
            date=date,
        )

        # Write file
        full_file_path.write_text(content)

        print(f"✅ Created file: {full_file_path}")
        print(f"✅ Used template: {template}")

    except Exception as e:
        print(f"❌ Error creating file {file_path}: {e}")
        sys.exit(1)


def pnm(module_name: str) -> None:
    """
    Package New Module: Create a complete module structure in src/

    Args:
        module_name: Name of the module to create
    """
    try:
        # Create module directory in src/
        module_dir = Path("src") / module_name
        module_dir.mkdir(parents=True, exist_ok=True)

        # Create __init__.py
        init_file = module_dir / "__init__.py"
        init_content = f'''"""
{module_name} module.

Description: {module_name} module
Author: SunnitAI
Date: 2024
"""
from src.registry import *

__version__ = "0.1.0"
__author__ = "SunnitAI"
__email__ = ""

# Import main classes/functions
# from .core import main_function
# from .utils import helper_function

__all__ = [
    # "main_function",
    # "helper_function",
]
'''
        init_file.write_text(init_content)

        # Create core.py
        core_file = module_dir / "core.py"
        core_content = f'''"""
Core functionality for {module_name}.

Description: Core functions and classes
Author: SunnitAI
Date: 2024
"""
from src.registry import *


def main_function():
    """Main function for {module_name}."""
    print(f"Running {module_name} main function")


if __name__ == "__main__":
    main_function()
'''
        core_file.write_text(core_content)

        # Create utils.py
        utils_file = module_dir / "utils.py"
        utils_content = f'''"""
Utility functions for {module_name}.

Description: Helper functions and utilities
Author: SunnitAI
Date: 2024
"""
from src.registry import *


def helper_function():
    """Helper function for {module_name}."""
    print(f"Running {module_name} helper function")
    return True


def validate_input(data):
    """Validate input data."""
    if not data:
        raise ValueError("Data cannot be empty")
    return True
'''
        utils_file.write_text(utils_content)

        # Create test file in src/test/
        test_dir = Path("src/test")
        test_dir.mkdir(parents=True, exist_ok=True)

        # Create __init__.py in test directory if it doesn't exist
        test_init = test_dir / "__init__.py"
        if not test_init.exists():
            test_init.write_text('"""Test package."""\n')

        test_file = test_dir / f"test_{module_name}.py"
        test_content = f'''"""
Tests for {module_name} module.

Description: Test cases for {module_name}
Author: SunnitAI
Date: 2024
"""
from src.registry import *

import unittest
from unittest.mock import patch, MagicMock
from {module_name}.core import main_function
from {module_name}.utils import helper_function, validate_input


class Test{module_name.title().replace("_", "")}(unittest.TestCase):
    """Test cases for {module_name} module."""
    
    def setUp(self):
        """Set up test fixtures."""
        pass
    
    def tearDown(self):
        """Tear down test fixtures."""
        pass
    
    def test_main_function(self):
        """Test main_function."""
        # Test that main_function runs without error
        try:
            main_function()
            self.assertTrue(True)
        except Exception as exc:
            self.fail(f"main_function raised {{exc}}")
    
    def test_helper_function(self):
        """Test helper_function."""
        result = helper_function()
        self.assertTrue(result)
    
    def test_validate_input_valid(self):
        """Test validate_input with valid data."""
        result = validate_input("test data")
        self.assertTrue(result)
    
    def test_validate_input_invalid(self):
        """Test validate_input with invalid data."""
        with self.assertRaises(ValueError):
            validate_input("")
        with self.assertRaises(ValueError):
            validate_input(None)


if __name__ == "__main__":
    unittest.main()
'''
        test_file.write_text(test_content)

        print(f"✅ Created module: {module_dir}")
        print(f"✅ Created __init__.py: {init_file}")
        print(f"✅ Created core.py: {core_file}")
        print(f"✅ Created utils.py: {utils_file}")
        print(f"✅ Created test file: {test_file}")

    except Exception as e:
        print(f"❌ Error creating module {module_name}: {e}")
        sys.exit(1)


def pnt(test_name: str, module_name: Optional[str] = None) -> None:
    """
    Package New Test: Create a test file in src/test/

    Args:
        test_name: Name of the test file to create (without test_ prefix)
        module_name: Optional module name to import in the test
    """
    try:
        # Create test directory
        test_dir = Path("src/test")
        test_dir.mkdir(parents=True, exist_ok=True)

        # Create __init__.py in test directory if it doesn't exist
        init_file = test_dir / "__init__.py"
        if not init_file.exists():
            init_file.write_text('"""Test package."""\n')

        # Create test file path
        test_path = test_dir / f"test_{test_name}.py"

        # Create test content with optional module import
        if module_name:
            import_line = f"from {module_name} import *"
            test_class_name = f"Test{module_name.title().replace('_', '')}"
        else:
            import_line = "# from your_module import your_function"
            test_class_name = f"Test{test_name.title().replace('_', '')}"

        test_content = f'''"""
Test module for {test_name}

Description: Test cases for {test_name}
Author: SunnitAI
Date: 2024
"""
from src.registry import *

import unittest
from unittest.mock import patch, MagicMock
{import_line}


class {test_class_name}(unittest.TestCase):
    """Test cases for {test_name}."""
    
    def setUp(self):
        """Set up test fixtures."""
        pass
    
    def tearDown(self):
        """Tear down test fixtures."""
        pass
    
    def test_example(self):
        """Test example."""
        self.assertTrue(True)


if __name__ == "__main__":
    unittest.main()
'''

        # Write test file
        test_path.write_text(test_content)

        print(f"✅ Test file created: {test_path}")

    except Exception as e:
        print(f"❌ Error creating test {test_name}: {e}")
        sys.exit(1)


# CLI wrapper functions for console scripts
def pni_cli():
    """CLI wrapper for pni function."""
    parser = argparse.ArgumentParser(
        description="Install packages and update pyproject.toml"
    )
    parser.add_argument("packages", nargs="+", help="Package names to install")
    parser.add_argument("--version", help="Version specification")

    args = parser.parse_args()

    # Install each package sequentially
    for package in args.packages:
        pni(package, args.version)


def pnd_cli():
    """CLI wrapper for pnd function."""
    parser = argparse.ArgumentParser(
        description="Create directory with __init__.py in src/"
    )
    parser.add_argument("path", help="Directory path to create (relative to src/)")
    parser.add_argument(
        "--test", action="store_true", help="Create in src/test/ instead of src/"
    )

    args = parser.parse_args()
    pnd(args.path, args.test)


def pnf_cli():
    """CLI wrapper for pnf function."""
    parser = argparse.ArgumentParser(
        description="Create Python file with template in src/"
    )
    parser.add_argument("file", help="File path to create (relative to src/)")
    parser.add_argument(
        "--template",
        default="basic",
        choices=["basic", "class", "cli", "test"],
        help="Template type",
    )
    parser.add_argument(
        "--test", action="store_true", help="Create in src/test/ instead of src/"
    )

    args = parser.parse_args()
    pnf(args.file, args.template, args.test)


def pnm_cli():
    """CLI wrapper for pnm function."""
    parser = argparse.ArgumentParser(description="Create complete module in src/")
    parser.add_argument("module", help="Module name to create")

    args = parser.parse_args()
    pnm(args.module)


def pnt_cli():
    """CLI wrapper for pnt function."""
    parser = argparse.ArgumentParser(description="Create test file in src/test/")
    parser.add_argument("test", help="Test name (without test_ prefix)")
    parser.add_argument("--module", help="Module name to import in test")

    args = parser.parse_args()
    pnt(args.test, args.module)


def start_dev_server(port: int = 8000, host: str = "127.0.0.1"):
    """Start the development server with uvicorn."""
    try:
        print(f"🚀 Starting development server on {host}:{port}...")
        cmd = [
            sys.executable,
            "-m",
            "uvicorn",
            "main:app",
            "--reload",
            "--host",
            host,
            "--port",
            str(port),
        ]
        subprocess.run(cmd, check=True)
    except subprocess.CalledProcessError as e:
        print(f"❌ Error starting development server: {e}")
        print("💡 Try using a different port with: pdev --port 8001")
        sys.exit(1)
    except KeyboardInterrupt:
        print("\n✅ Development server stopped")


def dev_cli():
    """CLI wrapper to start development server with uvicorn."""
    parser = argparse.ArgumentParser(
        description="Start development server with uvicorn"
    )
    parser.add_argument(
        "--port",
        type=int,
        default=8000,
        help="Port to run the server on (default: 8000)",
    )
    parser.add_argument(
        "--host",
        default="127.0.0.1",
        help="Host to bind the server to (default: 127.0.0.1)",
    )

    args = parser.parse_args()
    start_dev_server(args.port, args.host)


def main():
    """Main CLI entry point for interactive use."""
    parser = argparse.ArgumentParser(description="Python project scaffolding utilities")
    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    # pni subcommand
    pni_parser = subparsers.add_parser(
        "pni", help="Install packages and update pyproject.toml"
    )
    pni_parser.add_argument("packages", nargs="+", help="Package names to install")
    pni_parser.add_argument("--version", help="Version specification")

    # pnd subcommand
    pnd_parser = subparsers.add_parser(
        "pnd", help="Create directory with __init__.py in src/"
    )
    pnd_parser.add_argument("path", help="Directory path to create (relative to src/)")
    pnd_parser.add_argument(
        "--test", action="store_true", help="Create in src/test/ instead of src/"
    )

    # pnf subcommand
    pnf_parser = subparsers.add_parser(
        "pnf", help="Create Python file with template in src/"
    )
    pnf_parser.add_argument("file", help="File path to create (relative to src/)")
    pnf_parser.add_argument(
        "--template",
        default="basic",
        choices=["basic", "class", "cli", "test"],
        help="Template type",
    )
    pnf_parser.add_argument(
        "--test", action="store_true", help="Create in src/test/ instead of src/"
    )

    # pnm subcommand
    pnm_parser = subparsers.add_parser("pnm", help="Create complete module in src/")
    pnm_parser.add_argument("module", help="Module name to create")

    # pnt subcommand
    pnt_parser = subparsers.add_parser("pnt", help="Create test file in src/test/")
    pnt_parser.add_argument("test", help="Test name (without test_ prefix)")
    pnt_parser.add_argument("--module", help="Module name to import in test")

    # dev subcommand
    dev_parser = subparsers.add_parser(
        "dev", help="Start development server with uvicorn"
    )
    dev_parser.add_argument(
        "--port",
        type=int,
        default=8000,
        help="Port to run the server on (default: 8000)",
    )
    dev_parser.add_argument(
        "--host",
        default="127.0.0.1",
        help="Host to bind the server to (default: 127.0.0.1)",
    )

    args = parser.parse_args()

    if not args.command:
        parser.print_help()
        return

    if args.command == "pni":
        for package in args.packages:
            pni(package, args.version)
    elif args.command == "pnd":
        pnd(args.path, getattr(args, "test", False))
    elif args.command == "pnf":
        pnf(args.file, args.template, getattr(args, "test", False))
    elif args.command == "pnm":
        pnm(args.module)
    elif args.command == "pnt":
        pnt(args.test, getattr(args, "module", None))
    elif args.command == "dev":
        start_dev_server(
            getattr(args, "port", 8000), getattr(args, "host", "127.0.0.1")
        )


if __name__ == "__main__":
    main()
