# Contributing

## Development

To develop this library futher, it is recommended that you use Poetry for dependency management, virtualization and testing. You can read about how to install Poetry [here](https://python-poetry.org/docs/). One way for MacOS / Linux systems is

```bash
pip install pipx
pipx install poetry
pipx ensurepath # simply ensures that ~/.local/bin is added to $PATH
```

We also recommend creating the virtualenvs in the current directory itself

```bash
poetry config virtualenvs.in-project true
```

To install the dependencies and the project

```bash
poetry shell # starts a new shell inside the virtual environment
poetry install
```

`poetry install` by default installs the package in editable mode.

Create a .env file in the `compute-eval` directory.

```env
NEMO_API_KEY="<PUT-YOUR-KEY-HERE>"
```

or

```
API_KEY="<PUT-YOUR-KEY-HERE>"
```

if using a custom model.

### Linting

You will need to install the Black Python formatter and the Isort Formatter and lint on save. To do this in VSCode is simple, get [Black Python Formatter](https://marketplace.visualstudio.com/items?itemName=ms-python.black-formatter) and [Isort Formatter](https://marketplace.visualstudio.com/items?itemName=ms-python.isort) from the Marketplace
and then add these lines to either your workspace settings.json or your global settings.json

```json
"[python]": {
    "editor.defaultFormatter": "ms-python.black-formatter",
    "editor.formatOnSave": true,
    "editor.codeActionsOnSave": {
        "source.organizeImports": "always"
    },
},
"isort.args":["--profile", "black"],
```

Everytime you save the files, the linter will automatically lint for you. Depending on your workflow, you might want to have it check and report and then ask for permission to format the files.

## Sharing your contributions

For any additonal contributions that are made, please include a DCO in your commit message: https://wiki.linuxfoundation.org/dco
