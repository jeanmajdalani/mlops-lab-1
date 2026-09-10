# Lab 1 Answers

## Question 1
After running `uv init`, several files and folders are created:

- `.python-version`: specifies the Python version used by the project.
- `pyproject.toml`: contains project configuration, metadata, Python requirements, and dependencies.
- `README.md`: contains documentation about the project.
- `.gitignore`: specifies files and folders that Git should ignore.
- `src/`: contains the Python source code of the project.

These files create the basic structure of a Python project managed with uv.

## Question 2
Running `dvc init` creates DVC configuration files, mainly:

- `.dvc/config`: stores the DVC project configuration.
- `.dvc/.gitignore`: prevents internal DVC cache and temporary files from being tracked by Git.
- `.dvcignore`: tells DVC which files or directories it should ignore.
- `.dvc/tmp/`: contains temporary DVC files.

The configuration files such as `.dvc/config`, `.dvc/.gitignore`, and `.dvcignore`
should be pushed to Git.

Temporary files and cached dataset contents should not be pushed to Git.

## Question 3
When `--global` is used, the DVC credentials are stored in the user's global
DVC configuration outside the Git repository.

Other configuration scopes include:

- Repository/project configuration: `.dvc/config`
- Local configuration: `.dvc/config.local`
- Global configuration: user-level configuration
- System configuration: machine-level configuration

Credentials such as passwords or access tokens should NEVER be pushed to GitHub.

Only non-secret configuration, such as the DVC remote URL and default remote,
should be committed.

## Question 4
After running:

`dvc add data`

DVC adds the `data` directory to `.gitignore`.

This happens because the real dataset should not be tracked directly by Git.
Instead, Git tracks the small `data.dvc` pointer file while DVC manages the
actual dataset.

Therefore GitHub stores the code and DVC metadata, but not thousands of image
files.

## Question 5
Yes, DVC creates a file called `data.dvc`.

It is a small YAML metadata/pointer file. It contains information that identifies
the tracked data, such as:

- a hash/checksum of the dataset
- the path of the tracked data
- metadata such as size and/or number of files

It does not contain the actual images.

DVC uses this pointer to determine exactly which version of the dataset belongs
to a particular Git commit.

## Question 6
On GitHub, the source code is present, but the actual Food-11 image dataset is
not stored there.

GitHub contains the `data.dvc` pointer file that represents the version of the
dataset.

The actual data is uploaded using:

`dvc push`

and is stored in the configured DagsHub DVC remote.

Therefore:

- GitHub → code + configuration + DVC pointer
- DagsHub DVC storage → actual dataset

## Question 7
After cloning the GitHub repository into a completely new folder, the actual
dataset is not downloaded automatically.

Git only downloads the source code and the `data.dvc` pointer.

To retrieve the actual dataset from the DVC remote, run:

`dvc pull`

DVC reads `data.dvc` and downloads the corresponding data from DagsHub.

## Question 8
After checking out an older Git commit that contains the old version of
`data.dvc` and running:

`dvc checkout`

the newer folders:

- `food11_processed`
- `food11_processed_mini`

should disappear because the old `data.dvc` points to the previous version of
the data that contained only the earlier dataset state.

When switching back to the latest branch and running `dvc checkout` again,
the newer processed datasets should return.