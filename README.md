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



___
# Розподіл обов'язків 

## Яковенко Олена
Реалізація зчитування графа з csv-файлу, запис даних у вигяді матриці суміжності з оптимізацією збереження даних.
Перевірка графа на наявність Гамільтонового циклу.

### реалізація check_graph
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

### реалізація is_ham_cycle
Ця функція перевіряє граф на наявність Гамільтоновго циклу. Для її реалізації було розроблено ще дві додаткові функції: ham_cycle_util, is_safe
Робота:

    Пошук Гамільтонового шляху завжди починається з нульової вершини. 
    Функція повертає результат допоміжної функції ham_cycle_util, що рекурсивно шукає шлях.

Параметри:

    graph (list[list[int]]): Матриця суміжності графа.

Повертає:

    True: Якщо граф містить Гамільтонів цикл.
    False: Якщо граф не містить Гамільтонів цикл.

### реалізація ham_cycle_util
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

#### реалізація is_safe
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
Рефакторинг та візуалізація роботи алгоритму з використанням Pygame, імплементація роботи з конфіг-файлом.

## Гомбош Олег

## Кульбаба Андрій
Реалізація мурашиного алгритму

## Використанні технології
- Python 3.12
- NumPy 2.1.3
- pygame 2.6.1

___
## Імплементація 
___
## Results
___
