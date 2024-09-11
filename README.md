# Super Resolution
Image super resolution.


## Dependency
- python
    - [x] 3.7
- PyTorch
    - [x] 1.8.1
- torchvision
    - [x] 0.9.1


## Training datasets
- [x] freiburg groceries dataset
- [x] flicker1024
- [x] DIV2K
- [ ] shopee2020


## Run service
1. clone from git

    ```shell=
    $ git clone https://github.com/kaoyuching/super_resolution.git
    $ cd super_resolution
    ```

2. Install requirements
    - Install `pytorch` manually from [official website](https://pytorch.org/get-started/previous-versions/)
    - Install requirements.txt

        ```shell=
        $ pip install -r requirements.txt
        ```

2. Run
    - Run with user mode

        ```shell
        $ uvicorn main:app --host 0.0.0.0 --port 8000
        ```

    - Run with develop mode

        ```shell
        $ uvicorn main:app --host 0.0.0.0 --port 8000 --reload
        ```
