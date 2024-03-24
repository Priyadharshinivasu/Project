package com.tnsif.shoppingmall;

import java.util.List;

import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.web.bind.annotation.CrossOrigin;
import org.springframework.web.bind.annotation.DeleteMapping;
import org.springframework.web.bind.annotation.GetMapping;
import org.springframework.web.bind.annotation.PathVariable;
import org.springframework.web.bind.annotation.PostMapping;
import org.springframework.web.bind.annotation.PutMapping;
import org.springframework.web.bind.annotation.RequestBody;
import org.springframework.web.bind.annotation.RequestMapping;
import org.springframework.web.bind.annotation.RestController;

@RestController
@RequestMapping("/ShoppingMall")
@CrossOrigin(origins = "http://localhost:4200/")
public class ShoppingMallController {
	@Autowired
	private ShoppingMallRepository smRepo;
	
	@GetMapping("/{id}")
	public ShoppingMall getUserById(@PathVariable Long id)
	{
		return smRepo.findById(id).get();
	}
	
	@PostMapping
	public ShoppingMall createUser(@RequestBody ShoppingMall user)
	{
		return smRepo.save(user);
	}
	
	@PutMapping("/{id}")
	public ShoppingMall updateUser(@RequestBody ShoppingMall user,@PathVariable Long id)
	{
		user.setId(id);
		return smRepo.save(user); 
	}
	
	@DeleteMapping("/{id}")
	public void deleteUser(@PathVariable Long id)
	{
		smRepo.deleteById(id);
	}
	
	@GetMapping
	public List<ShoppingMall> getAllUser()
	{
		return smRepo.findAll();
	}
	
	@GetMapping("/Msg")
	public String getMsg()
	{
		return "This is my backend application";
		
	}

}
