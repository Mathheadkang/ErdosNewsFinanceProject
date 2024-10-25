# Contribution

Thank you for considering contributing to the Erdos News Finance Project! We welcome contributions from everyone. Please
awere the contribution princiles to help us maintain the code space.

## Principles
- Follow the existing code style.
- Keep the existing files structures.
- Write clear, concise commit messages.
- Include comments and documentation where necessary.

## Setup
Here
are some guidelines to help you get started:


1. **Python Version**: Make sure to use 3.13.0 under the project path.
	```bash
	# change the directory to your project path in the terminal
	# if you have multiple versions of python installed and the default one is not 3.13.0, you will need to specify it
	poetry env use <path>/python3.13
	# check the python version
	python --version
	# should be Python 3.13.0
	```

2. **Install Poetry**: We use [Poetry](https://python-poetry.org/) for dependency (python package version) management.
	```bash
	pip install poetry

	# check if successfully installed
	poetry -V
	```

3. **Install Python packages via Poetry**: all packages in the `pyproject.toml` file will be installed in the virtualenv
   path (Not in the project path. This feature could benefit for duplicated packaged files across different
   projects used with poetry).
	```bash
	poetry install --no-root

	# check if the packages has been successfully installed
	poetry show
	```

4. **Test the environment**

	You can test the environment by one of three following ways:

	(1) run python file directly
		```bash
		poetry run python tests/test_env/test_/test.py
		```

	(2) run the bash shell file which controled the python file
		```bash
		bash tests/test_env/run_test_env.sh 
		```

	(3) run test.py file in the Jupyter notebook `tests/test_env/test.ipynb`

	You can check [here](#configure-the-experiemnt-and-run-scripts) for more details.

## How to Contribute
1. **Clone the repostory to local** and change the directory to the project path.
	```bash
	cd <path>/ErdosNewsFinanceProject
	```
2. **Create a new branch**: 
	```bash
	git checkout -b your-branch-name
	```
3. **Pull the newest version from Github server**:
	```bash
	git pull
	```
3. **Make your changes**: Implement your feature or bug fix.
	If you need to install a new package, you will need to use Poetry to install it.
	```bash
	Poetry add <package>
	Poetry install
	```
4. **Commit your changes**: 
	```bash
	git commit -m "Description of your changes"
	```
6. **Push to your change to the Github server**: 
	```bash
	git push origin your-branch-name
	```
7. **Create a Pull Request**: When you finished all steps of a plan and want to merge your code with ours, please
   creat a Pull Resquest Go to the original repository in the Github webpage, click "New Pull Request" and **add the description** about your effort.

## Report Issues

	To report issues, please use the [GitHub Issues page](https://github.com/your-repo/ErdosNewsFinanceProject/issues). Provide a detailed description of the problem, steps to reproduce it, and any relevant logs or screenshots. This will help us address the issue more efficiently.

# File Structure

The project is organized as follows:

	```
	/ErdosNewsFinanceProject
	├── config/             # Experiement setup by yaml files
	├── src/                # Source code files
	│   ├── driver/         # Driver scripts to control pipelines in the engine
	│   ├── ingestion/      # Data ingestion
	│   ├── clean/          # Data preprocessing
	│   ├── model_fin/      # Model scripts for finance data
	│   ├── model_news/     # Model scripts for news data
	│   ├── evaluation/     # Model evaluation
	│   └── utils/              # Utility functions (reusable functions in the engine)
	├── scripts/            # Shell scripts to run engine driver
	├── notebook/           # Jupyter notebook for visualization or small experiments
	├── data/               # Data files
	│   ├── figures/        # Figure files
	│   └── models/         # Model files
	├── tests/              # Test files
	├── doc/                # Project documentation
	├── .gitignore          # Git ignore file
	├── README.md           # Project introduction
	├── pyproject.toml      # Poetry file to control dependents
	├── poetry.lock         # Poetry file
	└── CONTRIBUTING.md     # Contribution guidelines
	```

This structure helps in maintaining a clean and organized codebase, making it easier to navigate and contribute to the
project.File sturcture


# Configure the Experiemnt and Run Scripts

**Run Python scripts**
	```bash
	# method 1. run <file>.py script with poetry
	poetry run python <file>.py

	# method 2. open poetry shell
	poetry shell
	## then open python at the shell
	python
	## or run python scripts
	python <file>.py
	```

	If you want to run python code in the Jupyter Notebook, you will need to choose the kernel as the poetry environment for
	the current project first. For example, in VS Code, after open the Jupyter Notebook, you will find the bottom "Select
	Kernel" on the top right
	to choose the kernel.

# Style Guide