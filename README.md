### University of California, Berkeley

### Master of Information and Data Science Program (MIDS)

### DATASCI207 - Applied Machine Learning

Year: 2026 \
Semester: Spring \
Section: 5 \
Instructor: Nedelina Teneva \
Team Members: 4
* Alvaro Laham  
* Ruchi Tirumala  
* Jake Klein  
* Phillip Dundas

### Virtual Environment Setup
1. Open a terminal and change directories (using the `cd` command) into the project repo folder
2. Run the following command: `python -m venv venv`. This will create a folder (called `venv` in the project repo folder
3. Activate the virtual environment by running (for Mac) `source venv/bin/activate`, (for Windows) `venv\Scripts\activate`
4. Run `pip install -r requirements.txt`. This file `requirements.txt` has a list of all the software packages that are necessary (thus far) to run the project. The command installs all of those packages (and their dependent packages) into your venv.
5. From now on, when you develop, you should "use" this venv.
  - If you run things from the command line (ie terminal), activate the venv.
  - If you're running Jupyter notebooks (ie DataHub or an IDE), you'll need to make sure that the kernel you are using is using the venv (may be the wrong terminology here...). Ask Jake and he can help you do this - its sort of annoying.
6. If you import new packages that others will need to run your code, you should add those packages to the `requirements.txt` file. Then, others can update their virtual environment by simply running Step 4. 
