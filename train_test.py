import torch.utils.data.dataset


# Función que modela el entrenamiento de la red en cada epoch
def train(model, data_loader, optimizer, loss_criteria):
    model.train()   # seteamos el modelo en modo training
    train_loss = 0

    for batch, tensor in enumerate(data_loader):
        data, target = tensor
        # Se pasan los datos por la red y se calcula la función de loss
        optimizer.zero_grad()
        out = model(data)
        loss = loss_criteria(out, target)
        train_loss += loss.item()

        # Se hace la backpropagation y se actualizan los parámetros de la red
        loss.backward()
        optimizer.step()

    # Se devuelve el loss promedio
    avg_loss = train_loss / len(data_loader.dataset)
    return avg_loss


# Función que realiza el test de la red en cada epoch
def test(model, data_loader, loss_criteria):
    model.eval()  # seteamos el modelo en modo evaluación
    test_loss = 0
    correct = 0

    with torch.no_grad():
        for batch, tensor in enumerate(data_loader):
            data, target = tensor
            # Dado el dato, obtenemos la predicción
            out = model(data)

            # Calculamos el loss
            test_loss += loss_criteria(out, target).item()

            # Calculamos la accuracy (exactitud) (Sumando el resultado como
            # correcto si la predicción acertó)
            _, predicted = torch.max(out.data, 1)
            correct += torch.sum(target == predicted).item()

    # Devolvemos la exactitud y loss promedio
    avg_accuracy = correct / len(data_loader.dataset)
    avg_loss = test_loss / len(data_loader.dataset)
    return avg_loss, avg_accuracy


def dataloader(dataset, batch_size):
    return torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=0)


def print_data(percent, epoch, train_loss, test_loss, accuracy):
    print('{:.0f}% Epoch {:d}: loss entrenamiento={:.4f}, loss validación={:.4f}, exactitud={:.4%}'
          .format(percent, epoch, train_loss, test_loss, accuracy))
