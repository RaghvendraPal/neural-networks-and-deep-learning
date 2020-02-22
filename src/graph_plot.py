import matplotlib.pyplot as plt
import sys
class Graph:
    def accuracy_plot(self, accuracy, name_plot):
        # print(accuracy)
        plt.plot(accuracy)
        plt.title('model accuracy')
        plt.ylabel('Accuracy Plot')
        plt.xlabel('epoch')
        plt.legend(['test'], loc='upper left')
        plt.savefig(sys.path[0]+'/fig/'+name_plot+'.png')
        # plt.show()
        plt.close()


    def loss_plot(self, loss, name_plot):
        plt.plot(loss)
        plt.title('model loss')
        plt.ylabel('Loss Plot')
        plt.xlabel('epoch')
        plt.legend(['test'], loc='upper left')
        plt.savefig(sys.path[0]+'/fig/'+name_plot+'.png')
        # plt.show()
        plt.close()


    def cnn_n_factor_plot(self, accuracy, n_factors):
        plt.plot(n_factors,accuracy)
        plt.title('N Factors Accuracy')
        plt.ylabel('Accuracy')
        plt.xlabel('N Facors')
        plt.legend(['Train'], loc='upper left')
        plt.savefig(sys.path[0]+'/fig/n_factors_accuracy.png')
        # plt.show()
        plt.close()
