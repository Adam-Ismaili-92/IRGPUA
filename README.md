# Notre programme

IL FAUT mettre les images dans le dossier "images" (désolé c'est trop lourd pour le mail).

Afin de changer quel algorithme est utilisé dans notre programme,
il faut changer la fonction dans le `main.cu` indiqué par les options indiqué par le FIXME.
De la même manière avec un FIXME, vous pouvez choisir d'utiliser soit nôtre reduce, ou celui de thrust.

Notre algorithme compare systématiquement à la baseline CPU, comme ça on peut check à chaque fois que les images sont identiques.

# Setup pre-commit  🏗️

```shell
sudo apt-get install clang-format
pip install pre-commit
pre-commit install
```


# Enforce pre-commit to run 🏃

```shell
pre-commit run --all-files
```
