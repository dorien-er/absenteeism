# Data analysis
- Project: Absenteeism on the workfloor
- Description: Data Science/Analytics project integrating Python, SQL and Tableau. Multiple factors, including illness and transportation cost, can result in increased absenteeism during the work hours. In this project, machine learning is applied to predict increased absenteeism.
- Data Source: Absenteeism.CSV (from: https://www.udemy.com/course/python-sql-tableau-integrating-python-sql-and-tableau/)
- Type of analysis: Data preprocessing (Python/SQL), Data Viz (Tableau/Python), Machine Learning, Prediction Pipeline and Module building


# Startup the project

The initial setup.

Create virtualenv and install the project:
```bash
sudo apt-get install virtualenv python-pip python-dev
deactivate; virtualenv ~/venv ; source ~/venv/bin/activate ;\
    pip install pip -U; pip install -r requirements.txt
```

Unittest test:
```bash
make clean install
```

Check for absenteeism in gitlab.com/{group}.
If your project is not set please add it:

- Create a new project on `gitlab.com/{group}/absenteeism`
- Then populate it:

```bash
##   e.g. if group is "{group}" and project_name is "absenteeism"
git remote add origin git@github.com:{group}/absenteeism.git
git push -u origin master
git push -u origin --tags
```

# Install

Go to `https://github.com/{group}/absenteeism` to see the project, manage issues,
setup you ssh public key, ...

Create a python3 virtualenv and activate it:

```bash
sudo apt-get install virtualenv python-pip python-dev
deactivate; virtualenv -ppython3 ~/venv ; source ~/venv/bin/activate
```

Clone the project and install it:

```bash
git clone git@github.com:{group}/absenteeism.git
cd absenteeism
pip install -r requirements.txt
make clean install
                # install
```
