# Atomic-potential
ANPP — это нейронная сеть для расчёта потенциальной энергии молекул.
  Знание потенциальной энергии позволяют, рассчитать многие физические и химические свойства, она используется для поиска лекарств и материалов, и в фундаментальной науке. В общем нужная штука.
  В классическом случае, есть два похода для её расчета : квантовая химия и молекулярная механика.
  Квантовая химия рассчитывает энергию из взаимодействий электронов и ядер атома, с помощью уравнений квантовой механики и позволяет довольно точно рассчитать потенциальную энергию, но у её методов большая вычислительная стоимость, а сложность от N^1.5 до N^10, и, как правило, чем выше степень, тем точнее. Поэтому ситуации, когда один расчёт проводится на суперкомпьютере несколько месяцев или лет — это нормально.
  Во втором подходе, молекулярная механика, потенциальная энергия вычисляется из взаимодействий атомов с помощью физических параметризованных моделей, в ней предполагается, что ядра и электроны можно не брать в расчёт и рассчитывать взаимодействия только для атомов. Это позволяет уменьшить количество частиц + сами физические модели считаются быстрее квантовой механики и в итоге, расчёты быстрее в раз 10^5 и более. При этом с помощью неё можно получать довольно точные значения энергий, но тут проблема в параметризации модели, это может отнять много времени, при этом, заранее не известно, правильно ли ты запараметризавал, верные ли получились цифры, результаты нужно проверять.
  Вот и получается, что в первом случае — долго, во втором — много мороки.
  Альтернатива этим двум подходам нейронные сети. С помощью них можно быстро и точно вычислять, легко изменять форму функции и параметризовывать её. Так же можно воспользоваться ансамблем нейросетей с Гауссовским процессом и быть уверенным в расчетах, если ты уверен в данных, на которых обучаешь.
  Беллером и Парренинелло была разработана архитектура сети, в которой потенциальная энергия молекулы разбивается на сумму вкладов каждого атома в молекуле
