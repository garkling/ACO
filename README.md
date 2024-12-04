# ACO pathfinding algorithm

![391729413-05f9d95b-c9ef-4844-abef-33b1ea583896](https://github.com/user-attachments/assets/26115235-f06f-4681-9319-b673d9600b76)

## Огляд проблеми
### Модуль імплементує мурашиний алгоритм. 

Мурашиний алгоритм - один з ефективних поліноміальних алгоритмів для знаходження наближених розв'язків задачі комівояжера, а також аналогічних завдань пошуку маршрутів на графах. Підхід запропонований бельгійським дослідником Марко Доріго. Суть підходу полягає в аналізі та використанні моделі поведінки мурах, що шукають дороги від колонії до їжі.
**Результат корелює з кількістю ітерацій**, проте, з огляду на імовірність рішення, повторення алгоритму може видавати досить точний результат. 

### Де використовується мурашиний алгоритм?
Наша команда вибрала тему мурашиного алгоритму, адже він є досить цікавим та практичним.
Мурашиний алгоритм застосовується для розв'язання задач оптимізації в різних галузях:

- **Транспорт і логістика:** Задача комівояжера, маршрутизація вантажів, оптимізація графіків транспорту.
- **Мережі:** Оптимізація маршрутизації в комп'ютерних і бездротових мережах.
- **Розклад і планування:** Планування робочих змін, розклад завдань у хмарних обчисленнях.
- **Біоінформатика:** Вирівнювання послідовностей ДНК, моделювання білків.
- **Робототехніка:** Координація роботів і прокладання маршрутів.
- **Фінанси:** Оптимізація інвестиційних портфелів.

### Принцип роботи
У основі алгоритму лежить поведінка мурашиної колонії — маркування вдалих доріг великою кількістю феромону. Робота починається з розміщення мурашок у вершинах графу (містах), потім починається рух мурашок — напрям визначається імовірнісним методом, на підставі формули:

```math
P_i = \frac{l_i^q \cdot f_i^p} {\sum_{k=0}^N l_k^q \cdot f_k^p}
```

де:

- $P_i$ — ймовірність переходу шляхом $i$,
- $l_i$ — величина, обернена до довжини (ваги) $i$-ого переходу,
- $f_i$ — кількість феромонів на $i$-ому переході,
- $q$ — величина, яка визначає «жадібність» алгоритму,
- $p$ — величина, яка визначає «стадність» алгоритму, і
- $q + p = 1$

Пройдений мурахою шлях відображається, коли мураха відвідає всі вузли графу. Петлі виключено, оскільки в алгоритм включено список табу. Після завершення довжина шляху може бути підрахована — вона дорівнює сумі довжин всіх ребер, якими подорожувала мураха. Рівняння (1) показує кількість феромону, який був залишений на кожному ребрі шляху для мурашки k. Змінна Q є константою.

```math
\Delta \tau_{ij}^k(t) = \frac{Q}{L^k(t)} 
```

Результат рівняння є засобом вимірювання шляху, — довжина шлюху є обернено-пропорційна до концентрації феромонів на ньому. Далі, отриманий результат використовується в рівнянні (2), щоб збільшити кількість феромону вздовж кожного ребра пройденого мурахою шляху.

```math
\tau_{ij}(t) = \Delta \tau_{ij}(t) + (\tau_{ij}^k \times \rho) 
```

Важливо, що дане рівняння застосовується до всього шляху, при цьому кожне ребро позначається феромоном обернено-пропорційно до довжини шляху. Тому слід дочекатися, поки мураха закінчить подорож і лише потім оновити рівні феромону, в іншому випадку справжня довжина шляху залишиться невідомою. Константа p — значення між 0 і 1. 
![image](https://github.com/user-attachments/assets/167b03b0-da98-442b-945b-d80a96202c89)

# Запуск
Змінити налаштування:
- `nano config.ini`
   
Змінити дані графа:
- `nano graph.csv`
   
В папці з проєктом запустити для обрахунку:
- `./run.sh`
    
В папці з проєктом запустити для візуалізації:
(потрібно перейти на гілку `visualization` через несумісність `pygame` з `multiprocessing`)
- `./visualize.sh`

___
# Розподіл обов'язків 

## Яковенко Олена
Реалізація зчитування графа з csv-файлу, запис даних у вигяді матриці суміжності з оптимізацією збереження даних.
Перевірка графа на наявність Гамільтонового циклу.

## реалізація check_graph
Ця функція обробляє дані про граф з csv-файлу, та перетворює їх на матрицю суміжності.

Оптимізація:

    Відбувається перейменування вершин графа, кожна вершина має свій номер з діапазону [0, n-1].
    Це забезпечує уникнення порожніх рядків і стовпців, якщо вершини мають розріджену або невпорядковану нумерацію.

Параметри:

    filename (str): Шлях до файлу з даними про граф.

Повертає:

    list[list[int]]: Матриця суміжності, що описує з’єднання між вершинами.
                     - матриця є квадратною та симетричною, адже граф є неорієнтованим
                     - елементи на діагоналі — нулі
                     - у відповідній комірці матриці записується вага цього ребра між вершинами,
                       якщо ребра немає, то записується значення дорівнює нулю


## реалізація is_ham_cycle
Ця функція перевіряє граф на наявність Гамільтоновго циклу. Для її реалізації було розроблено ще дві додаткові функції: ham_cycle_util, is_safe

Робота:

    Пошук Гамільтонового шляху завжди починається з нульової вершини. 
    Функція повертає результат допоміжної функції ham_cycle_util, що рекурсивно шукає шлях.

Параметри:

    graph (list[list[int]]): Матриця суміжності графа.

Повертає:

    True: Якщо граф містить Гамільтонів цикл.
    False: Якщо граф не містить Гамільтонів цикл.


## реалізація ham_cycle_util
Ця функція перевіряє граф на наявність Гамільтоновго шляху.

Алгоритм:

    Функція здійснює рекурсивний пошук гамільтонового циклу, додаючи до шляху одну вершину за раз і перевіряючи можливість повернення 
    до початкової вершини після відвідування всіх вершин. У процесі пошуку перевіряється кожна вершина, і якщо вона може бути додана 
    до шляху, функція продовжує пошук для наступної вершини. Якщо після завершення шляху вдалося повернутися до початкової вершини 
    і всі вершини були відвідані, гамільтоновий цикл існує.
    
    Базовий випадок:
    Коли всі вершини вже додані до шляху, перевіряється наявність ребра між останньою вершиною в шляху і початковою, щоб завершити цикл.

    Рекурсія:
    Для кожної вершини графа виконується перевірка її безпечності для додавання до поточного шляху:
      Використовується функція is_safe, яка перевіряє:
        - Чи існує ребро між попередньою вершиною в шляху і поточною.
        - Чи не була ця вершина вже додана до шляху.
      Якщо вершина задовольняє умови, вона додається до шляху, і алгоритм переходить до наступної позиції.
      У разі відсутності рішення виконується "відкат" (backtracking).
    
    Відкат:
    Якщо вершину не можна додати до шляху, вона видаляється, і алгоритм продовжує спроби з іншими вершинами.

Параметри:

    graph (list[list[int]]): Матриця суміжності графа.
    path (list[int]): Список, що містить поточний маршрут.
    pos (int): Поточна позиція в шляху, на яку намагаються додати вершину.

Повертає:

    True: Якщо граф містить Гамільтонів цикл.
    False: Якщо граф не містить Гамільтонів цикл.


## реалізація is_safe

Робота:

    Функція виконує перевірку чи можна додати вершину v до поточного шляху в позиції pos, а саме:
       - Чи існує ребро між попередньою вершиною в шляху і поточною.
       - Чи не була ця вершина вже додана до шляху.

Параметри:

    v (int): Вершина, яку потрібно перевірити для додавання до шляху.
    graph (list[list[int]]): Матриця суміжності графа.
    path (list[int]): Поточний шлях, що будується.
    pos (int): Поточна позиція в шляху, де потрібно додати вершину v.

Повертає:

    True: Якщо вершина v може бути додана до шляху на позицію pos.
    False: Якщо вершина не може бути додана, оскільки порушуються умови для гамільтонового шляху.



## Бобрик Владислав
Рефакторинг та візуалізація роботи алгоритму з використанням `Pygame`, імплементація роботи з конфіг-файлом.
### реалізація `visualization.py`
Модуль призначений для візуалізації маршруту, знайденого за допомогою алгоритму мурашиної оптимізації, на випадково згенерованому повному графі.

Алгоритм:

    Головна функція ініціалізовує матриці стану мурах на екрані, генерує випадковий повний граф для заданої кількості вузлів. 
    Після цього запускається алгоритм та збираються дані про шлях мурах на кожній ітерації. 
    Дані використовуються для відмальовки кожної мурахи на шляху з однієї ноди до іншої. 
    Прогрес пройденого шляху та швидкості мурах, а також поточна ітерація та вихідна нода мурахи - все це зберігається в матрицях і передається при кожному виклику `move_ants`.
    
    Функція `draw_graph` приймає набір нод та їх коородинати з дистанціями і відмальовує на екрані. 
    Також функція відмальовує кожне ребро різним кольором від світло-сірого до чорного, залежно від частоти відвідування цього ребра мурахами.

    Малюванння відбувається завдяки бібліотеці `pygame`, яка надає високорівневий інтерфейс екрану для малювання, а також методи оновлення екрану
    та годинник для задання частоти оновлень екрану.

### Конфіг-файл
Невеличка модифікація для зручного задання всіх параметрів алгоритму, використовуючи `configparser` та `config.ini` файл.  
Всі параметри задаються в окремому файлі у відповідних секціях, після чого програма парсить і використовує їх під час роботи алгоритму.

## Кульбаба Андрій і Гомбош Олег
###Реалізація мурашиного алгритму

### реалізація run_iteration
Ця функція виконує одну ітерацію алгоритму мурашиної оптимізації для знаходження маршруту в графі.

Алгоритм:

    Ініціалізація мурашок:
        Для кожної мурашки створюється локальна видимість, яка змінюється в міру відвідування вузлів.

    Побудова маршруту:
        Для кожного вузла в маршруті:
            Поточний вузол позначається як відвіданий.
            Обчислюються характеристики вузлів:
                Феромонна складова: залежить від концентрації феромонів.
                Видимість: обернено пропорційна відстані.
                Остаточна характеристика вузла є добутком цих складових.
            Ймовірність переходу до наступного вузла визначається за допомогою накопичувальної суми нормалізованих характеристик.
            Випадкове число визначає, до якого вузла переходить мурашка.
    
    Завершення маршруту:
        Останній вузол обирається таким чином, щоб маршрут містив усі вузли графа.

Параметри:

    graph (ndarray): Матриця суміжності графа.
    visibility (ndarray): Матриця видимості (обернені відстані).
    pheromone (ndarray): Матриця феромонів.
    params (object): Об’єкт із параметрами алгоритму.
    node_num (int): Кількість вузлів у графі.
    path (ndarray): Маршрут, який будується для кожної мурашки.

Повертає:

    path (ndarray): Оновлений маршрут для кожної мурашки.


### реалізація worker_run_iteration
Ця функція є обгорткою для паралельного запуску run_iteration.

Робота:

    Приймає всі параметри алгоритму як єдиний кортеж.
    Викликає run_iteration з розпакованими параметрами.

Параметри:

    args (tuple): Кортеж параметрів: graph, visibility, pheromone, params, node_num, path.

Повертає:

    Результат виконання run_iteration.


### реалізація update_pheromones
Ця функція оновлює матрицю феромонів на основі маршрутів мурашок і зменшує концентрацію феромонів через випаровування.

Алгоритм:

    Випаровування:
    Зменшує феромони на всіх ребрах пропорційно коефіцієнту випаровування.

    Оновлення феромонів:
        Для кожного маршруту обчислюється його загальна довжина.
        Для кожного ребра у маршруті додається феромонна складова, обернено пропорційна довжині маршруту.

Параметри:

    pheromone (ndarray): Матриця феромонів.
    path (ndarray): Шляхи, побудовані мурашками.
    graph (ndarray): Матриця суміжності графа.
    params (object): Об’єкт із параметрами алгоритму.
    node_num (int): Кількість вузлів у графі.

Повертає:

    pheromone (ndarray): Оновлена матриця феромонів.
    tour_total_distance (ndarray): Массив із загальною довжиною маршрутів для кожної мурашки.


### реалізація find_best_path
Ця функція знаходить найкращий маршрут і його довжину серед усіх маршрутів мурашок.

Робота:

    Визначається індекс маршруту з мінімальною довжиною.

Параметри:

    path (ndarray): Маршрути мурашок.
    tour_total_distance (ndarray): Загальна довжина маршрутів.

Повертає:

    best_path (ndarray): Найкращий маршрут.
    best_distance (float): Довжина найкращого маршруту.


### реалізація run_ant_colony_optimization
Ця функція реалізує алгоритм оптимізації мурашиним колоніями для пошуку найкращого шляху в графі.

Алгоритм:

    Ініціалізація: 
        Функція починає з ініціалізації параметрів графа, кількості вершин, видимості та феромонів, а також початкових шляхів мурашок.

    Робота з пулом процесів:
        Алгоритм розділяє роботу на кілька процесів для паралельного виконання за допомогою ProcessPoolExecutor. Кожен процес виконує окремі ітерації алгоритму для мурашок.
        Кількість ітерацій визначається на основі кількості ітерацій в параметрах, і кожна ітерація обробляється у групах по 4 процеси.
    
    Оновлення феромонів:
        Після кожної ітерації відбувається оновлення феромонів на основі знайдених шляхів. Паралельно обчислюються відстані для кожного шляху.
    
    Пошук найкращого шляху:
        Після кожної ітерації порівнюється поточний найкращий шлях з новим, і у разі покращення результату оновлюється кращий шлях.
    
    Остаточне оновлення відстані:
        Після завершення всіх ітерацій додається відстань до початкової вершини для замкнення шляху.
    
Параметри:

    params (object): Об'єкт, який містить параметри алгоритму, включаючи кількість мурашок, кількість ітерацій, швидкість випаровування феромонів та інші налаштування.

Повертає:

    best_path (list[int]): Найкращий знайдений шлях серед усіх мурашок.
    best_distance (int): Відстань найкращого шляху.


## Використанні технології
- Python 3.12
- NumPy 2.1.3
- pygame 2.6.1

___
## Results

Наша програма дозволяє користувачам задавати граф через CSV-файл, де кожен рядок містить інформацію про дві вершини та вагу між ними. Програма використовує алгоритм мурашиної колонії для пошуку найкоротшого шляху, що проходить через кожну вершину лише один раз, якщо такий шлях існує. 
Наша реалізація дозволяє ефективно розв'язувати задачу пошуку Гамільтонового шляху у графах різної складності. Ми оптимізували цей алгоритм завдяки паралелелізму, використовуючи високорівнену бібліотеку concurrent, розподіляємо наші ітерації між 4 процесами, після чого шукаєм найкращі шляхи серед них 4. Завдяки цьому код працює набагато швидше, особливо на великих наборах данних.

___
```text
5 вершин - 4.96 сек
graph = [
    [0, 2, 9, 10, 1],
    [2, 0, 6, 4, 8],
    [9, 6, 0, 3, 7],
    [10, 4, 3, 0, 5],
    [1, 8, 7, 5, 0]
]
Path: [1. 5. 4. 3. 2. 1.]
Distance: 17



10 вершин - 8.85 сек

graph = [
    [0, 4, 8, 10, 3, 9, 7, 5, 6, 2],
    [4, 0, 2, 7, 3, 5, 6, 8, 10, 4],
    [8, 2, 0, 4, 9, 1, 7, 6, 5, 3],
    [10, 7, 4, 0, 8, 5, 2, 9, 6, 1],
    [3, 3, 9, 8, 0, 6, 4, 7, 5, 10],
    [9, 5, 1, 5, 6, 0, 8, 3, 2, 4],
    [7, 6, 7, 2, 4, 8, 0, 9, 10, 3],
    [5, 8, 6, 9, 7, 3, 9, 0, 1, 4],
    [6, 10, 5, 6, 5, 2, 10, 1, 0, 8],
    [2, 4, 3, 1, 10, 4, 3, 4, 8, 0]
]
Path: [ 1. 10.  4.  7.  5.  2.  3.  6.  8.  9.  1.]
Distance: 25


15 вершин - 19.94 с
graph = [
    [0, 6, 8, 10, 12, 5, 9, 3, 4, 7, 6, 8, 9, 2, 4],
    [6, 0, 5, 9, 7, 4, 6, 10, 8, 3, 2, 5, 7, 9, 6],
    [8, 5, 0, 4, 6, 10, 8, 3, 2, 7, 9, 6, 4, 8, 5],
    [10, 9, 4, 0, 7, 6, 5, 8, 9, 3, 2, 6, 7, 4, 8],
    [12, 7, 6, 7, 0, 8, 5, 4, 10, 6, 9, 3, 8, 7, 9],
    [5, 4, 10, 6, 8, 0, 9, 2, 7, 4, 6, 5, 8, 9, 7],
    [9, 6, 7, 2, 4, 8, 0, 4, 7, 10, 6, 8, 9, 3, 4],
    [3, 10, 3, 8, 4, 2, 4, 0, 5, 6, 9, 7, 6, 5, 2],
    [4, 8, 2, 9, 10, 7, 7, 5, 0, 3, 4, 9, 5, 8, 6],
    [7, 3, 7, 3, 6, 4, 10, 6, 3, 0, 2, 5, 8, 9, 4],
    [6, 2, 9, 2, 9, 6, 6, 9, 4, 2, 0, 8, 7, 5, 6],
    [8, 5, 6, 6, 3, 5, 8, 7, 9, 5, 8, 0, 4, 7, 9],
    [9, 7, 4, 7, 8, 8, 9, 6, 5, 8, 7, 4, 0, 5, 8],
    [2, 9, 8, 4, 7, 9, 3, 5, 8, 9, 5, 7, 5, 0, 4],
    [4, 6, 5, 8, 9, 7, 4, 2, 6, 4, 6, 9, 8, 4, 0]
]

Path: [ 1. 14.  7. 13. 12. 15.  6.  2. 11.  4. 10.  9.  3.  8.  5.  1.]
Distance: 55
```
