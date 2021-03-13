# 根据指定的宽度打印格式良好的价格列表 P47
width = 45#width = int(input('Please enter width: '))
price_width = 10
item_width = width - price_width
header_fmt1 = '{{:{}}}{{:>{}}}'.format(item_width, price_width)
fmt1 = '{{:{}}}{{:>{}.2f}}'.format(item_width, price_width)
print('{{:{}}}  {{:>{}.2f}}'.format(item_width, price_width) + '消除了两对{}，这个是当存在格式内部也为{}时，即{:{}}的固定用法')
print('=' * width)
print(header_fmt1.format('Item', 'Price'))
print('-' * width)
print(fmt1.format('Apples', 0.4))  #继续替换剩下的一对{}
print(fmt1.format('Pears', 0.5))
print(fmt1.format('Cantaloupes', 1.92))
print(fmt1.format('Dried Apricots (16 oz.)', 8))
print(fmt1.format('Prunes (4 lbs.)', 12))
print('=' * width)
print("居中文本".center(width))