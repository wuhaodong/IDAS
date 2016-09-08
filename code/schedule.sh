#WP_DIR=/home/wu75/Scheduling/wp
#WEB_DIR=/home/wu75/public_html
#LOG_DIR=/home/leo820209/log
#SC_DIR=/home/leo820209/scheduling

#cd $WP_DIR
#wp export --allow-root --path=$WEB_DIR/wordpress --filename_format=AFY.xml --dir=$WP_DIR --post_type=attachment
python wp.py AFY.xml
#cp $WP_DIR/data.json $SC_DIR/data.json
#cp $WP_DIR/data.json $WEB_DIR/data.json

