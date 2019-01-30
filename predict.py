# Imports

from utils import loading_checkpoint, map_category_names, process_image
import torch
import argparse

def processing_arguments():

    parser = argparse.ArgumentParser()
    parser.add_argument("image_path", help="Specifies the image location", default="flower_data/test/28/image_05230.jpg")
    parser.add_argument('checkpoint', help="Checkpoint file", default="checkpoint.pth")
    parser.add_argument("--top_k", type=int,  help="Specifies the top K most likely classes", default=1)
    parser.add_argument('--category_names', help="Specifies the file of the category names")
    parser.add_argument("--gpu",help="Specifies the use of gpu", action="store_true")

    args = parser.parse_args()

    image_path = args.image_path
    checkpoint = args.checkpoint
    top_k = args.top_k
    if args.category_names:
        category_names = args.category_names
    else:
        category_names = None

    if args.gpu and torch.cuda.is_available():
        device = "cuda"
    else:
        device = "cpu"

    print(args)
    return image_path, checkpoint, top_k, category_names, device

def predict(image_path, model, topk, device):
    model.eval()
    model.to(device)
    np_im = process_image(image_path)

    # Converting it to Tensor
    im = torch.from_numpy(np_im).float()
    if device == 'cuda': im = im.cuda()
    im = im.unsqueeze(0)

    with torch.no_grad():
        output = model.forward(im)

    ps = torch.exp(output)

    probs, indices = ps.topk(topk)

    #Cannot convert CUDA tensor to numpy. So, copying the tensor to host memory first.
    indices = indices.cpu()
    probs = probs.cpu()

    # inverting dictionary
    inv_map = {v: k for k, v in model.class_to_idx.items()}
    # mapping
    classes = list()
    for label in indices.numpy()[0]:
        classes.append(inv_map[label])

    return probs.numpy()[0], classes


def main():

    # Obtaining command line arguments
    image_path, checkpoint, top_k, category_names, device = processing_arguments()
    #Loading the trained model
    model = loading_checkpoint(checkpoint)
    #print(model)
    #Class prediction
    probs, classes = predict(image_path, model, top_k, device)

    print ("Image:", image_path)
    print ("Probs\tClass")

    #If category_names flag is specified, then we print the name of the classes instead
    if category_names:
        classes = map_category_names(category_names,classes)

    for label, prob in zip (classes,probs):
            print ("{:.3f}".format(prob),"\t{}".format(label))


if __name__ == '__main__':
    main()
