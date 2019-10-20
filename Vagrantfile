# $Env:VAGRANT_PREFER_SYSTEM_BIN += 0

Vagrant.configure("2") do |config|
	config.vm.box = "ubuntu/xenial64"
	config.vm.synced_folder ".", "/vagrant", :disabled => true
	config.vm.provider "virtualbox" do |vb|
		vb.memory = 2048
		vb.cpus = 2
	end
	
	config.vm.provision :shell, :path => "Scripts/Setup.sh"
	config.vm.synced_folder "Code", "/home/Code"
	
end