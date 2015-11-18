from multiprocessing import Pool


def func(inp):
    print(inp * 3)
    return inp

if __name__ == '__main__':
    pool = Pool(processes=5)
    pages = pool.map(func, ['a','b','c','d', 'e','f','h','i','j','k','l','m','n','o','p','q','r','s','t','u','v','w','x','y','z'])

    print(pages)
