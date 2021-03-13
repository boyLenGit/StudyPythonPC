import matplotlib.pyplot as plt


def learn1():
    foo = ['a', 'b', 'c']
    bar = [1, 2, 3]
    fig, axes = plt.subplots(2, 2, figsize=(6, 6))  # 第一个参数是创建的figure
    axes[0][0].bar(foo, bar)
    axes[0][1].scatter(bar, bar)
    axes[1][0].plot(bar, bar)
    axes[1][1].pie(bar)
    plt.show()

def learn2():
    X = [[1, 2, 1, 7], [3, 4, 2, 10], [5, 1, 1, 5]]
    plt.imshow(X)
    plt.colorbar()
    plt.show()


if __name__ == '__main__':
    learn2()
