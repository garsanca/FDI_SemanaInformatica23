#include <CL/sycl.hpp>
#include <vector>


int main() {
	std::cout << "List Platforms and Devices" << std::endl;
	std::vector<sycl::platform> platforms = sycl::platform::get_platforms();
	for (const auto &plat : platforms) {
		// get_info is a template. So we pass the type as an `arguments`.
		std::cout << "Platform: "
			<< plat.get_info<sycl::info::platform::name>() << " "
			<< plat.get_info<sycl::info::platform::vendor>() << " "
			<< plat.get_info<sycl::info::platform::version>() << std::endl;





		std::vector<sycl::device> devices = plat.get_devices();
		for (const auto &dev : devices) {
			std::cout << "Device: "
				<< dev.get_info<sycl::info::device::name>() << std::endl
				<< "     is host?=" << (dev.is_host() ? "Yes" : " No") << std::endl
				<< "     is gpu?=" << (dev.is_gpu() ? "Yes" : " No") << std::endl
				<< "     Driver version " << dev.get_info<sycl::info::device::driver_version>() << std::endl
				<< "     Vendor ID " << dev.get_info<sycl::info::device::vendor_id>() << std::endl
				<< "     Compute Units " << dev.get_info<sycl::info::device::max_compute_units>() << std::endl
				<< "     Max Work-Group size is " << dev.get_info<sycl::info::device::max_work_group_size>() << std::endl
				<< "     Max Global mem size is " << dev.get_info<sycl::info::device::global_mem_size>() << std::endl
				<< "     Max Local mem size is " << dev.get_info<sycl::info::device::local_mem_size>() << std::endl
				<< std::endl;
			}
	}
}
