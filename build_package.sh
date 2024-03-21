#!/bin/bash

# Install krb5-devel package (or the equivalent for your distribution)
sudo apt-get install -y krb5-devel

# Install Python modules
sudo pip3 install -r requirements.txt | sed -e 's/^/install_log=/'

# Create after-install script
echo '#!/bin/bash
systemctl enable process_anomaly_package.service
systemctl start process_anomaly_package.service' > after-install.sh
chmod +x after-install.sh

# Create the package
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

# Clean up after-install script
rm after-install.sh