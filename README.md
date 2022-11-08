# Super Resolution
Image super resolution.


## Dependency
- python
    - [x] 3.7
- PyTorch
    - [x] 1.8.1
- torchvision
    - [x] 0.9.1

## Run service
1. clone from git

    ```shell=
    $ git clone http://10.0.4.52:3000/doris_kao/super_resolution.git
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
