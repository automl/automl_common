# automl_common
This repository holds shared utilities that AutoML frameworks may benefit from.

It is not meant to be used standalone, and each of the modules available in this repository
contains individual guidelines on how to use best make use of them.


## Installation
We recommend installing this repository via git submodule as follows:

```
git submodule add git@github.com:automl/automl_common.git <AutoMLFramework>/automl_common
```

This will create 2 new files in your project repository, being `.gitmodules` and `<AutoMLFramework>/automl_common`. You can then commit the changes (`git commit -am "<msg>"`) and push them to your repository (`git push origin master`). Further information can be seen [here](https://git-scm.com/book/en/v2/Git-Tools-Submodules).

Here are some additional tips when using this repository:

When cloning your `<AutoMLFramework>`, you have to initialize also the submodule, and this can be done in two ways:

```
git clone --recurse-submodules <url to AutoMLFramework>
```


```
git clone <url to AutoMLFramework>
cd AutoMLFramework
git submodule update --init --recursive
```


## Contributing
The modules available in this repository comply with pep-8 style guidelines.
Please create a Pull-Request to the main branch, and ensure the code correctness through:
```
pre-commit run --all-files
python -m pytest test
```


## Contact

This repository is developed by the [AutoML Group of the University of Freiburg](http://www.automl.org/).
