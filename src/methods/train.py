import architectures.siamese_network as siamese_network
import torch
import utils
import numpy as np


def train(cnfg):
    network = siamese_network.FeatureNetwork()
    training_cnfg = cnfg["TRAINING"]
    device = training_cnfg["DEVICE"]
    parameters = training_cnfg["PARAMETERS"]
    # network.to(device)
    images, _, labels, _ = utils.load_data("data")

    labels = torch.from_numpy(labels)
    # dataset = utils.gen_triplet_dataset(labels, parameters['BATCH_SIZE'], parameters['STEPS_PER_EPOCH'])
    epochs = parameters["EPOCHS"]
    batch_size = parameters["BATCH_SIZE"]
    batches_per_epoch = int(np.floor(len(images) / batch_size))

    loss_fn = torch.nn.TripletMarginLoss()
    optimizer = torch.optim.Adam(
        network.parameters(), lr=parameters["LR"], betas=(0.9, 0.99), eps=1e-7
    )
    for epoch in range(epochs):
        epoch_permutation = torch.randperm(len(images))

        epoch_images = [images[i] for i in epoch_permutation]
        epoch_labels = labels[epoch_permutation]
        losses = []
        epoch_loss = 0
        for batch in range(batches_per_epoch):
            batch_images = epoch_images[batch * batch_size : (batch + 1) * batch_size]
            batch_labels = epoch_labels[batch * batch_size : (batch + 1) * batch_size]
            optimizer.zero_grad()
            embeddings = []
            for image in batch_images:
                embedding = network(image.reshape(1, 3, 299, 299))[0, 0]
                embeddings.append(embedding)
            # print(embeddings)
            embeddings = torch.cat(embeddings, dim=0)
            loss, _ = _batch_all_triplet_loss(
                batch_labels, embeddings, 2.0, device, True
            )
            loss += batch_hard_triplet_loss(batch_labels, embeddings, 2.0, device, True)
            loss.backward()
            optimizer.step()
            print(
                f"epoch {epoch+1}/{epochs}, batch {batch+1}/{batches_per_epoch} Loss: {loss}"
            )
        # plt.plot(range(len(losses)),losses)
        # plt.show()

    torch.save(network, "saved.model")
    network = torch.load("saved.model")


def _pairwise_distances(embeddings, device, squared=False):
    transposed_embeddings = torch.transpose(embeddings, 0, 1)
    dot_product = torch.matmul(embeddings, transposed_embeddings)
    square_norm = torch.diag(dot_product)
    distances = (
        torch.unsqueeze(square_norm, 0)
        - 2.0 * dot_product
        + torch.unsqueeze(square_norm, 1)
    )

    distances = torch.maximum(
        distances, torch.Tensor([0.0]).expand_as(distances).to(device=device)
    )

    if not squared:
        mask = torch.eq(distances, 0.0).type(torch.float)
        distances = distances + mask * 1e-16

        distances = torch.sqrt(distances)
        distances = distances * (1.0 - mask)

    return distances


def batch_hard_triplet_loss(labels, embeddings, margin, device, squared=False):
    # Get the pairwise distance matrix
    pairwise_dist = _pairwise_distances(embeddings, device, squared=squared)

    # For each anchor, get the hardest positive
    # First, we need to get a mask for every valid positive (they should have same label)
    mask_anchor_positive = _get_anchor_positive_triplet_mask(labels)
    mask_anchor_positive = mask_anchor_positive.type(torch.float)

    # We put to 0 any element where (a, p) is not valid (valid if a != p and label(a) == label(p))
    anchor_positive_dist = torch.multiply(mask_anchor_positive, pairwise_dist)

    # shape (batch_size, 1)
    hardest_positive_dist = torch.max(anchor_positive_dist, dim=1, keepdim=True).values

    # For each anchor, get the hardest negative
    # First, we need to get a mask for every valid negative (they should have different labels)
    mask_anchor_negative = _get_anchor_negative_triplet_mask(labels)
    mask_anchor_negative = mask_anchor_negative.type(torch.float)

    # We add the maximum value in each row to the invalid negatives (label(a) == label(n))
    max_anchor_negative_dist = torch.max(pairwise_dist, dim=1, keepdim=True).values
    anchor_negative_dist = pairwise_dist + max_anchor_negative_dist * (
        1.0 - mask_anchor_negative
    )

    # shape (batch_size,)
    hardest_negative_dist = torch.min(anchor_negative_dist, dim=1, keepdim=True).values

    # Combine biggest d(a, p) and smallest d(a, n) into final triplet loss
    loss = hardest_positive_dist - hardest_negative_dist + margin
    triplet_loss = torch.maximum(
        loss, torch.Tensor([0.0]).expand_as(loss).to(device=device)
    )

    # Get final mean triplet loss
    triplet_loss = torch.mean(triplet_loss)

    return triplet_loss


def _batch_all_triplet_loss(labels, embeddings, margin, device, squared=False):
    pairwise_dist = _pairwise_distances(embeddings, device, squared=squared)

    anchor_positive_dist = torch.unsqueeze(pairwise_dist, 2)
    anchor_negative_dist = torch.unsqueeze(pairwise_dist, 1)

    triplet_loss = anchor_positive_dist - anchor_negative_dist + margin

    mask = _get_triplet_mask(labels).type(torch.float)
    triplet_loss = torch.multiply(mask, triplet_loss)

    triplet_loss = torch.maximum(
        triplet_loss, torch.Tensor([0.0]).expand_as(triplet_loss).to(device=device)
    )

    valid_triplets = torch.greater(triplet_loss, 1e-16)
    num_positive_triplets = torch.sum(valid_triplets)
    num_valid_triplets = torch.sum(mask)
    fraction_positive_triplets = num_positive_triplets / (num_valid_triplets + 1e-16)

    # Get final mean triplet loss over the positive valid triplets
    triplet_loss = torch.sum(triplet_loss) / (num_positive_triplets + 1e-16)

    return triplet_loss, fraction_positive_triplets


def _get_triplet_mask(labels):
    masks = torch.ones((len(labels), len(labels), len(labels)))
    for i, a in enumerate(labels):
        for j, p in enumerate(labels):
            for k, n in enumerate(labels):
                isInvalidTriplet = a != p or n == a or i == j
                masks[i, j, k] = 0 if isInvalidTriplet else 1
    return masks.to("cuda")


def _get_anchor_positive_triplet_mask(labels):
    masks = torch.ones(len(labels), len(labels))
    for i, a in enumerate(labels):
        for j, p in enumerate(labels):
            isValid = a == p and i != j
            masks[i, j] = 1 if isValid else 0
    return masks.to("cuda")


def _get_anchor_negative_triplet_mask(labels):
    masks = torch.ones(len(labels), len(labels))
    for i, a in enumerate(labels):
        for j, n in enumerate(labels):
            isValid = a != n
            masks[i, j] = 1 if isValid else 0
    return masks.to("cuda")
