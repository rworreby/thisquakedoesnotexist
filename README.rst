=========================
This Quake Does Not Exist
=========================


.. image:: https://img.shields.io/pypi/v/thisquakedoesnotexist.svg
        :target: https://pypi.python.org/pypi/thisquakedoesnotexist

.. image:: https://img.shields.io/travis/rworreby/thisquakedoesnotexist.svg
        :target: https://travis-ci.com/rworreby/thisquakedoesnotexist

.. image:: https://readthedocs.org/projects/thisquakedoesnotexist/badge/?version=latest
        :target: https://thisquakedoesnotexist.readthedocs.io/en/latest/?version=latest
        :alt: Documentation Status




Synthetic Earthquake Data Generation using Generative Adverserial Networks


* Free software: MIT license
* Documentation: https://thisquakedoesnotexist.readthedocs.io.

Installation
------------
.. code-block:: bash

    # Create your virtual environment and activate it
    $ python3 -m venv venv
    $ source venv/bin/activate

    # Update pip and install dependencies
    $ pip install --upgrade pip
    $ pip install -r requirements.txt
    $ pip install -e .


MLFLow
------
All experiments are by default tracked by MLFlow, with the tracking URI default set to `/home/rworreby/thisquakedoesnotexist/mlruns/`.
Don't forget set the tracking URI to the local setup.
If you haven't set permissions for your user to create folders in that path, you might have to create the folder over the terminal first, using the following command:

.. code-block:: bash

    mkdir -p /home/mlflow/mlruns

Running Code
------------
The code is set up as follows:

1. Specifying what do to with `make`, e.g. train a model
2. Setting the model, parameters, and metadata with a `bash` script
3. Feed the info from point 2. into the code via `Argparse`
4. Run the code with the specified settings


Running on Sisma
----------------
One caveat of running on Sisma is that make and other services can time out if you don't detach the process from your user / connection.
In order to do this, start a process with `nohup` (or alternatively use `tmux` and detach), however, `nohup` is recommended. 
You should always pipe the error out put and std output if you detach a process.

Here is a full example of how to start a training process from a file and pipe the process:

.. code-block:: bash

    nohup make train_file file=run_amplitudes.sh &> train_log.txt &

Note that the parameter file `run_amplitudes.sh` must be places in the appropriate folder `thisquakedoesnotexist/runfiles/`
You can also check `make help` for a help on the `make` commands.


Features
--------

* TODO

Credits
-------

This package was created with Cookiecutter_ and the `audreyr/cookiecutter-pypackage`_ project template.

.. _Cookiecutter: https://github.com/audreyr/cookiecutter
.. _`audreyr/cookiecutter-pypackage`: https://github.com/audreyr/cookiecutter-pypackage
