build:
	bundle exec jekyll build -d ./html

serve:
	bundle exec jekyll serve

clean:
	bundle exec jekyll clean

push:
	rsync -P html root@192.168.50.242:/var/www
