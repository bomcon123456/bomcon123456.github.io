build:
	bundle exec jekyll build -d ./html

serve:
	bundle exec jekyll serve

clean:
	bundle exec jekyll clean

push:
	rsync -aP html root@192.168.50.242:/var/www/

make bpush:
	bundle exec jekyll build -d ./html
	rsync -aP html root@192.168.50.242:/var/www/

