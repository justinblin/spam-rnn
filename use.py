import torch
import preprocess
import postprocess

def use(rnn, data:str, labels_unique:list[str], threshold:float = 0.5) -> tuple[str, int]:
    rnn.eval()
    with torch.no_grad():
        data_tensors = preprocess.string_to_tensor(data)
        output_tensor = rnn.forward(data_tensors, show = True)
        guess, guess_index = postprocess.label_from_output(output_tensor, labels_unique, threshold)
        return guess, guess_index

def main():
    print(torch.__version__)

    # use GPU if possible
    device = torch.device('cpu')
    if torch.cuda.is_available():
        device = torch.device('cuda')
    torch.set_default_device(device)

    print(device)

    # get the list of unique labels
    labels_unique = ['ham', 'spam']

    # load the trained model
    rnn = torch.load('./my_model_512_3x_linear_leakyReLU', weights_only = False)
    rnn.to(device)
    print(rnn)

    # use the model
    example_data = "GENT! We are trying to contact you. Last weekends draw shows that you won a �1000 prize GUARANTEED. Call 09064012160. Claim Code K52. Valid 12hrs only. 150ppm"
    guess, guess_index = use(rnn, example_data, labels_unique)
    print(f'My RNN guessed "{example_data}" is {guess}')

if __name__ == "__main__":
    main()