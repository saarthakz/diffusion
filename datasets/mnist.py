from torchvision import transforms, datasets


def get_dataset(input_res: list[int], mean: list[int], std: list[int]):
    transform = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Resize(size=input_res),
            transforms.Normalize(
                mean=mean,
                std=std,
            ),
        ]
    )

    dataset = datasets.MNIST(
        root="./data",
        download=True,
        transform=transform,
    )

    return dataset
