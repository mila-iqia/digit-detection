# Submitting your projects

In order to ensure a streamlined and easy project submission, please follow these guidelines. Please note that failure to follow these guidelines can result in a submission that will not be accepted.

We have also opened [issues](https://github.com/mila-iqia/ift6759/issues) on github. This is a great place to ask your questions, that way others can see responses as well. You can also come talk to us during class and office hours or send us emails.

## How to submit

You will be submitting your entire projects on the Helios cluster. We have created a repository for you to submit your code. The repository is at

`/rap/jvb-000-aa/COURS2019/etudiants/submissions/`

Inside the `submissions/` folder, you must create a folder with your team name (i.e. `b1pdndt1`). There should be one and only one submission per team. Within the folder, we expect to find your report (.pdf format only!), a `model/` directory and a `code/` directory. Your folder structure should look something like this:

```
COURS2019/etudiants/submissions/
└── b1pdndt1
    ├── code
    │   ├── .git/
    │   └── evaluation
    │       ├── run_evaluation.sh
    │       └── eval.py
    ├── model
    │   └── best_model.pth
    └── sample_report.pdf

```

Your `code` repository should be a clone of the code on your master branch of the github repo. It should only contain code that is relevant to the evaluation. This is what will be used by other teams in the peer-review evaluation section of the project. In order to rename a folder to `code` when using `git clone`, you can use

`git clone https://github.com/username/your-project code`

from within your team directory. Your best model(s) and necessary files that should be shared to others for evaluation should go in the `model` directory. It is your responsibility to ensure that your code runs appropriately with respect to the model saved in that path.

You should ensure that your project is read-only by other groups using

`chmod -R g+r,g-w,o-r <path_to_team_submission>`

## Running your code

### Block 1 and 2
We will be running the code on our own test sets. It is your responsiblity to ensure that the code can be properly run based on the [`eval.py`](https://github.com/mila-iqia/digit-detection/tree/master/project/evaluation/code/eval.py) and the [`run_evaluation.sh`](https://github.com/mila-iqia/digit-detection/tree/master/project/evaluation/code/run_evaluation.sh) files that were provided to you.

We expect you to have completed the `eval.py` file and ensure that its outputs values are as expected.

We recommend you test your submissions prior to the deadline. Validate your methods with instructors prior to submission if necessary.

Keep in mind that the instructors will be running your code, so be sure that it does not depend on local files or environment variables that instructors would otherwise not be able to access.

To be clear, the code will be run by the instructors directly on the cluster **FROM YOUR SUBMISSION FOLDER ON HELIOS** using a test set that is not directly visible to you on the cluster. We will be using the `run_evaluation.sh` scripts on your submissions and expect them to work with no intervention on our end.

### Block 3
Please refer to the [block 3 evaluation instructions](https://github.com/mila-iqia/digit-detection/tree/master/project/evaluation/code/block3_evaluation_instructions.md)
