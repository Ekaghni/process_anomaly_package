#!/bin/bash


sudo apt-get install -y krb5-devel

sudo pip3 install -r requirements.txt | sed -e 's/^/install_log=/'

echo '#!/bin/bash
systemctl enable process_anomaly_package.service
systemctl start process_anomaly_package.service' > after-install.sh
chmod +x after-install.sh


fpm -s dir -t deb \
    -n process_anomaly_package \
    -v 1.0 \
    -d python3 \
    --directories /home/telaverge/process_anomaly_package/src \
    --config-files /etc/systemd/system/process_anomaly_package.service \
    --after-install after-install.sh \
    --python-install-data /usr/local/lib/python3.10/site-packages \
    src/=/home/telaverge/process_anomaly_package/src \
    process_anomaly_package.service=/etc/systemd/system/process_anomaly_package.service
rm after-install.sh