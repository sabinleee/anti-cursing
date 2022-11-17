# train.py 를 실행해서 model의 weight&bias를 만들어야합니다.
# 시간이 오래 걸립니다.
# 아니면 그냥 .pt를 그대로 사용할 수도 있음

from setuptools import setup, find_packages

setup(
    name="anti-cursing",   # pypi 에 등록할 라이브러리 이름
    version="0.0.1",    # pypi 에 등록할 version (수정할 때마다 version up을 해줘야 함)
    description="The package that detect & switch the curse word in the sentence by using deep learning",
    author="24_bean",
    author_email="sabin5105@gmail.com",
    url="https://github.com/sabin5105/anti-cursing",
    python_requires=">= 3.8",
    packages=find_packages(),
    install_requires=[],
    zip_safe=False,
    package_data={},
    include_package_data=True
)