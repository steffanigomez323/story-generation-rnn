fname = 'xfilesalloriginal.txt'
fname2 = 'xfilesfilesseason1and2.txt'
files = ['asongoffireandice.txt', 'edgarallenpoe.txt', 'finneganswake.txt', 'lolita.txt', 'nightvale.txt',
         'quentintarantino.txt', 'xfilesalloriginal.txt']

for file in files:

    data = []

    with open(file, "r") as fp:
        for line in fp:
            line = line.strip()
            if line[:8] != "file:///":
                line = line.decode('utf-8', 'ignore').encode("utf-8")
                data.append(line)

    with open(file, "w") as wp:
        for line in data:
            wp.write(line + "\n")