from glob import glob

from setuptools import find_packages, setup


package_name = "vision_avoid"

setup(
    name=package_name,
    version="0.1.0",
    packages=find_packages(exclude=["test"]),
    data_files=[
        ("share/ament_index/resource_index/packages", [f"resource/{package_name}"]),
        (f"share/{package_name}", ["package.xml"]),
        (f"share/{package_name}/config", glob("config/*.yaml")),
        (f"share/{package_name}/launch", glob("launch/*.py")),
        (f"share/{package_name}/urdf", glob("urdf/*")),
        (f"share/{package_name}/worlds", glob("worlds/*.world")),
    ],
    install_requires=["setuptools"],
    extras_require={"test": ["pytest<9"]},
    zip_safe=True,
    maintainer="Vision Avoid",
    maintainer_email="local@example.invalid",
    description="Single-camera local obstacle avoidance and scene safety gate",
    license="Apache-2.0",
    entry_points={
        "console_scripts": [
            "vision_camera_node = vision_avoid.camera_node:main",
            "vision_debug_web = vision_avoid.debug_web:main",
            "vision_stm32_bridge = vision_avoid.stm32_bridge:main",
            "vision_avoid_node = vision_avoid.node:main",
            "vision-download-models = vision_avoid.download_models:main",
            "vision-calibrate-camera = vision_avoid.calibrate_camera:main",
        ],
    },
)
