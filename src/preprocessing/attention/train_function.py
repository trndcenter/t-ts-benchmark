import torch
from tqdm import tqdm


def train(
    epochs, model, optimizer, num_train_commits, batch_size, dataset, margin, device, step: int = 10
):
    last_loss = None
    for epoch in tqdm(range(epochs), desc="Epochs"):
        model.train()
        print(f"Epoch: {epoch}")
        running_loss = 0
        iterations = 0
        for i in range(num_train_commits):
            item = dataset[i]
            # if there is less, then batch_size files in a commit, skip this commit
            if item["file_tokens_padded"].size(0) < batch_size:
                continue

            model, optimizer, running_loss, iterations = train_one_commit(
                item,
                model,
                optimizer,
                running_loss,
                iterations,
                batch_size,
                device,
                margin,
            )
            if i % step == step - 1:
                last_loss = running_loss / iterations
                print(f"{i + 1} batch loss: {last_loss:.3f}")
                running_loss = 0
                iterations = 0
        if last_loss:
            print(f"Loss train: {last_loss:.3f}.")

    return model


def train_one_commit(
    item,
    model,
    optimizer,
    running_loss,
    iterations,
    batch_size,
    device,
    margin,
    train=True,
):
    # Creating mini-batches on each commit
    for i_file in range(0, item["file_tokens_padded"].size(0) // batch_size):
        file_tokens_padded = item["file_tokens_padded"][
            i_file * batch_size : i_file * batch_size + batch_size
        ].to(device)
        file_lengths = item["file_lengths"][
            i_file * batch_size : i_file * batch_size + batch_size
        ].to(device)

        for i_test in range(0, item["pos_test_tokens_padded"].size(0) // batch_size):
            pos_test_tokens_padded = item["pos_test_tokens_padded"][
                i_test * batch_size : i_test * batch_size + batch_size
            ].to(device)
            pos_test_lengths = item["pos_test_lengths"][
                i_test * batch_size : i_test * batch_size + batch_size
            ].to(device)

            neg_test_tokens_padded = item["neg_test_tokens_padded"][
                i_test * batch_size : i_test * batch_size + batch_size
            ].to(device)
            neg_test_lengths = item["neg_test_lengths"][
                i_test * batch_size : i_test * batch_size + batch_size
            ].to(device)

            optimizer.zero_grad()

            pos_tests = model(
                file_tokens_padded,
                file_lengths,
                pos_test_tokens_padded,
                pos_test_lengths,
            )
            neg_tests = model(
                file_tokens_padded,
                file_lengths,
                neg_test_tokens_padded,
                neg_test_lengths,
            )

            loss = hinge_loss(
                torch.sigmoid(pos_tests), torch.sigmoid(neg_tests), torch.tensor(margin)
            )
            if train:
                loss.backward()
                optimizer.step()
            running_loss += loss.item()
            iterations += 1
    return model, optimizer, running_loss, iterations


def hinge_loss(pos_tests, neg_tests, hinge_margin=torch.tensor(0.5)):
    return torch.mean(
        (hinge_margin - (pos_tests - neg_tests))
        * ((hinge_margin - (pos_tests - neg_tests)) > 0).float()
    )
