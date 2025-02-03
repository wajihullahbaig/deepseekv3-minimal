import matplotlib.pyplot as plt

def plot_expert_activation(expert_activations):
    plt.bar(range(len(expert_activations)), expert_activations)
    plt.xlabel('Expert Index')
    plt.ylabel('Activation Count')
    plt.savefig('expert_activation.png')
    plt.show()