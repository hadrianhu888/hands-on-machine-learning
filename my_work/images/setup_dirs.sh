# bash script to generate the missing directories from chapter 3 to chapter 19

# create the directories

for i in {3..19}; do
    mkdir -p chapter$i
done

# create the files

for i in {3..19}; do
    touch chapter$i/README.md
done
